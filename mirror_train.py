import argparse
import os
import numpy as np
import math

from collections import deque


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


from generators import generators
from discriminators import discriminators
from siren import siren
import fid_evaluation

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy

from torch_ema import ExponentialMovingAverage


def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def train(opt):
    torch.manual_seed(0)

    device = torch.device(0)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z = z_sampler((25, 256), device='cpu', dist=metadata['z_dist'])

    SIREN = getattr(siren, metadata['model'])
    CHANNELS = 3
    scaler = torch.cuda.amp.GradScaler()

    if opt.load_dir != '':
        generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
        discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)
        ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
    else:
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
        discriminator = getattr(discriminators, metadata['discriminator'])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator.siren.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator.named_parameters() if n in mapping_network_param_names]
        generator_parameters = [p for n, p in generator.named_parameters() if n not in mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*5e-2}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_G.pth')))
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_D.pth')))
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scaler.pth')))

    generator_losses = []
    generator_sym_losses = []
    discriminator_losses = []
    max_memories = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(0)
    dataloader = None
    total_progress_bar = tqdm(total=opt.n_epochs, desc="Total progress", dynamic_ncols=True)  # Keeps track of total training
    total_progress_bar.update(discriminator.epoch)  # Keeps track of progress to next stage
    interior_step_bar = tqdm(dynamic_ncols=True)
    for _ in range(opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)
        
        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader, CHANNELS = datasets.get_dataset(metadata['dataset'], **metadata)

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)


            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        for i, (imgs, _) in enumerate(dataloader):
            if discriminator.step % opt.model_save_interval == 0:
                now = datetime.now()
                now = now.strftime("%d--%H:%M--")
                now = now + '_' + str(discriminator.step) + '--'
                torch.save(ema, os.path.join(opt.output_dir, now + 'ema.pth'))
                torch.save(ema2, os.path.join(opt.output_dir, now + 'ema2.pth'))
                torch.save(generator, os.path.join(opt.output_dir, now + 'generator.pth'))
                torch.save(discriminator, os.path.join(opt.output_dir, now + 'discriminator.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_G.pth'))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_D.pth'))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, now + 'scaler.pth'))
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata['batch_size']:
                break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator.train()
            discriminator.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
                    split_batch_size = z.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        g_imgs, g_pos = generator(subset_z, **metadata)

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                r_preds, _, _ = discriminator(real_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_latent, g_pred_position = discriminator(gen_imgs, alpha, **metadata)
                latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']
                position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                identity_penalty = latent_penalty + position_penalty

                d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
                discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)


            # TRAIN GENERATOR
            z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])

            split_batch_size = z.shape[0] // metadata['batch_split']

            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    generator.mirror_mode()
                    gen_imgs, gen_positions, gen_raw_point_predictions = generator(subset_z, **metadata)
                    generator.normal_mode()
                    #
                    # gen_imgs = gen_imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
                    # print(np.min(gen_imgs), np.max(gen_imgs))
                    # gen_imgs = ((gen_imgs + 1) / 2) * 255
                    # print(np.min(gen_imgs), np.max(gen_imgs))
                    # gen_imgs = gen_imgs.astype(np.uint8)
                    # from PIL import Image
                    # for img in gen_imgs:
                    #
                    #     print(img.shape)
                    #     im = Image.fromarray(img, 'RGB')
                    #     im.show()
                    #
                    # exit()
                    g_preds, g_pred_latent, g_pred_position = discriminator(gen_imgs, alpha, **metadata)

                    topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                    g_preds = torch.topk(g_preds, topk_num, dim=0).values
                    latent_penalty = torch.nn.MSELoss()(g_pred_latent, subset_z) * metadata['z_lambda']
                    position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_penalty = latent_penalty + position_penalty

                    g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty
                    generator_losses.append(g_loss.item())

                    # Symmetrical loss
                    half_split_batch_size = split_batch_size // 2
                    target_tensor = torch.cat((gen_raw_point_predictions[half_split_batch_size:, :], gen_raw_point_predictions[:half_split_batch_size, :]), dim=0)
                    if not opt.optimize_symmetric_color:
                        target_tensor[:, :3] = gen_raw_point_predictions[:, :3]
                    target_tensor = target_tensor.detach()
                    sym_loss = torch.nn.MSELoss()(gen_raw_point_predictions, target_tensor) * .5
                    generator_sym_losses.append(sym_loss.item())

                scaler.scale(g_loss + sym_loss * metadata.get('sym_lambda', 0)).backward()

            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator.parameters())
            ema2.update(generator.parameters())

            try:
                interior_step_bar.update(1)
            except Exception as e:
                pass
                # print('Error from interior_step_bar.update at line 304', e)
            if i % 10 == 0:
                try:
                    tqdm.write(f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()} (sym: {sym_loss.item()})] [Step: {discriminator.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]")
                except OverflowError:
                    pass
                    # print("Overflow error from line 307 in train.py")
            if discriminator.step % opt.sample_interval == 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        copied_metadata['img_size'] = 128
                        gen_imgs = generator.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed.png"), nrow=5, normalize=True)

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        copied_metadata['h_mean'] += math.pi / 2
                        copied_metadata['img_size'] = 128
                        gen_imgs = generator.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_tilted.png"), nrow=5, normalize=True)

            if (discriminator.step + 1) % 100 == 0:
                torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                torch.save(generator, os.path.join(opt.output_dir, 'generator.pth'))
                torch.save(discriminator, os.path.join(opt.output_dir, 'discriminator.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                torch.save(generator_sym_losses, os.path.join(opt.output_dir, 'generator_sym.losses'))
                torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))
                torch.save(max_memories, os.path.join(opt.output_dir, 'max_memories.sizes'))

            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')

                fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir, target_size=128)
                fid_evaluation.output_images(generator, metadata, 0, 1, generated_dir)
                fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, target_size=128)
                with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                    f.write(f'\n{discriminator.step}:{fid}')

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1
        max_memories.append(torch.cuda.max_memory_allocated(device=device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=1500)
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=1500)
    parser.add_argument('--optimize_symmetric_color', action='store_true', default=False)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)

    train(opt)
