import os
import shutil
import torch

from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
from pytorch_fid import fid_score
import datasets
from tqdm import tqdm
import copy
import argparse
import curriculums


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1

def setup_evaluation(dataset_name, generated_dir, target_size=128):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, img_size=target_size)
        print('outputting real images...')
        output_real_images(dataloader, 50, real_dir)
        print('...done')

    os.makedirs(generated_dir, exist_ok=True)
    return real_dir

def output_images(generator, input_metadata, rank, world_size, output_dir, num_imgs=2048):
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = 128
    metadata['batch_size'] = 4

    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 1

    img_counter = rank
    generator.eval()
    img_counter = rank

    if rank == 0: pbar = tqdm("generating images", total = num_imgs)
    with torch.no_grad():
        while img_counter < num_imgs:
            z = torch.randn((metadata['batch_size'], generator.z_dim), device=generator.device)
            generated_imgs, _ = generator.staged_forward(z, **metadata)

            for img in generated_imgs:
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
                img_counter += world_size
                if rank == 0: pbar.update(world_size)
    if rank == 0: pbar.close()

def calculate_fid(dataset_name, generated_dir, target_size=256):
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 64, 'cuda', 2048)
    torch.cuda.empty_cache()

    return fid

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('generator_file', type=str)
#     parser.add_argument('discriminator_file', type=str)
    parser.add_argument('output_dir', type=str, default='temp')
    parser.add_argument('curriculum', type=str)
    parser.add_argument('--num_images', type=int, default=2048)
    parser.add_argument('--gpu_type', type=str, default='8000')
    parser.add_argument('--keep_percentage', type=float, default='1.0')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--max_batch_size', type=int, default=None)

    opt = parser.parse_args()

    if '6' in opt.gpu_type:
        max_batch_size = 2400000
    else:
        max_batch_size = 94800000

    if opt.max_batch_size != None:
        max_batch_size = opt.max_batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curriculum = getattr(curriculums, opt.curriculum)

    real_images_dir = setup_evaluation(curriculum['dataset'], opt.output_dir, target_size=curriculum['img_size'])

    os.makedirs(opt.output_dir, exist_ok=True)

    generator = torch.load(opt.generator_file, map_location=device)
    generator.set_device(device)
    generator.eval()

    if opt.ema:
        curriculum['ema'] = True
    if curriculum['ema']:
        ema_file = opt.generator_file.split('generator')[0] + 'ema.pth'
        ema = torch.load(ema_file)
        ema.copy_to(generator.parameters())

    discriminator_file = opt.generator_file.split('generator')[0] + 'discriminator.pth'

    discriminator = torch.load(discriminator_file, map_location=device)
    discriminator.to(device)
    discriminator.eval()

    discriminator_scores = []
    discriminator_batch_size = 64
    discriminator_batch = []
    discriminator_batch_index = []

    assert(opt.num_images % discriminator_batch_size == 0)
    for img_counter in tqdm(range(opt.num_images)):
        z = torch.randn(1, 256, device=device)

        with torch.no_grad():
            img = generator.staged_forward(z, max_batch_size=max_batch_size, **curriculum)[0].to(device)
            save_image(img, os.path.join(opt.output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))

    metrics_dict = calculate_metrics(opt.output_dir, real_images_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
    print(metrics_dict)
