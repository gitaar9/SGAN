import argparse
import math
import os
from pathlib import Path

from torchvision.utils import save_image

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import skvideo.io
import curriculums
from torchvision import transforms


def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


def get_ground_truth_images(image_folder, n_input_views):
    img_names = ['000000.png', '000001.png', '000002.png', '000003.png'][:n_input_views]
    img_paths = [os.path.join(image_folder, "rgb", name) for name in img_names]

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256),
         transforms.Resize((opt.image_size, opt.image_size), interpolation=0), transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    gt_images = []
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img).to(device).unsqueeze(0)
        gt_images.append(img)

    return torch.cat(gt_images, dim=0)


def get_average_frequencies(generator):
    z = torch.randn((10000, 256), device=device)
    with torch.no_grad():
        frequencies, phase_shifts = generator.siren.mapping_network(z)
    w_frequencies = frequencies.mean(0, keepdim=True)
    w_phase_shifts = phase_shifts.mean(0, keepdim=True)
    return w_frequencies, w_phase_shifts


def generate_additional_frames(generator, render_options, num_frames, frequencies, phase_shifts, lock_view_dependence):
    trajectory = []
    for t in np.linspace(0, 1, num_frames):  # Turn around the boat with shifting pitch
        pitch = ((math.pi / 2 * 85 / 90) * t)
        yaw = (2 * math.pi * t) - (math.pi * 0.5)
        trajectory.append((pitch, yaw))
    for t in np.linspace(0, 1, num_frames):  # Turn around the boat at lowest level pitch
        pitch = (math.pi / 2 * 85 / 90)
        yaw = (2 * math.pi * t) - (math.pi * 0.5)
        trajectory.append((pitch, yaw))

    additional_frames = []
    with torch.no_grad():
        for pitch, yaw in tqdm(trajectory):
            render_options['h_mean'] = yaw
            render_options['v_mean'] = pitch
            with torch.cuda.amp.autocast():
                frame, depth_map = generator.staged_forward_with_frequencies(frequencies,
                                                                             phase_shifts,
                                                                             max_batch_size=opt.max_batch_size,
                                                                             lock_view_dependence=lock_view_dependence,
                                                                             **render_options)
                additional_frames.append(tensor_to_PIL(frame))
                # depths.append(tensor_to_PIL(depth_map))
    return additional_frames


def create_gif(frames, output_dir):
    output_name = 'inverse_render.gif'
    img, *imgs = frames
    img.save(fp=os.path.join(f'{output_dir}', output_name), format='GIF', append_images=imgs,
             save_all=True, duration=45, loop=0, interlace=False)


def create_mp4(frames, output_dir):
    output_name = 'inverse_render.avi'
    writer = skvideo.io.FFmpegWriter(os.path.join(f'{output_dir}', output_name),
                                     outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})

    frame_repeat = 2
    for frame in frames:
        for _ in range(frame_repeat):
            writer.writeFrame(np.array(frame))
    # for depth in depths:
    #     writer.writeFrame(np.array(depth))
    writer.close()


def main(opt, device):
    generator = torch.load(opt.generator_path, map_location=torch.device(device))
    generator.set_device(device)
    generator.eval()

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.manual_seed(0)

    lock_view_dependence = True

    # Load one or multiple images
    gt_images = get_ground_truth_images(opt.image_path, opt.n_input_views)

    # Inference poses:
    img_yaws = [math.pi / 2, math.pi, -math.pi/2, -math.pi/2]
    img_pitches = [math.radians(35), math.radians(35), math.radians(35), math.radians(85)]

    image_h_means = [-yaw_from_renderer + (math.pi * 0.75) - (0.5 * math.pi) for yaw_from_renderer in img_yaws]
    # For new no mirror generator add this:
    # image_h_means = [h - (math.pi / 100) * 64 for h in image_h_means]
    image_v_means = [(math.pi / 2 * 85 / 90) - pitch_from_renderer for pitch_from_renderer in img_pitches]

    gt_h_means = image_h_means[:opt.n_input_views]
    gt_v_means = image_v_means[:opt.n_input_views]

    options = {
        'img_size': opt.image_size,
        'fov': 30,
        'ray_start': 0.75,
        'ray_end': 1.25,
        'num_steps': 30,
        'h_stddev': 0,
        'v_stddev': 0,
        'hierarchical_sample': True,
        'sample_dist': None,
        'clamp_mode': 'relu',
        'nerf_noise': 0,
        'white_back': True
    }

    render_options = {
        'img_size': 128,
        'fov': 30,
        'ray_start': 0.75,
        'ray_end': 1.25,
        'num_steps': 30,
        'h_stddev': 0,
        'v_stddev': 0,
        'hierarchical_sample': True,
        'sample_dist': None,
        'clamp_mode': 'relu',
        'nerf_noise': 0,
        'white_back': True
    }

    w_frequencies, w_phase_shifts = get_average_frequencies(generator)

    w_frequency_offsets = torch.zeros_like(w_frequencies)
    w_phase_shift_offsets = torch.zeros_like(w_phase_shifts)

    w_frequency_offsets.requires_grad_()
    w_phase_shift_offsets.requires_grad_()

    optimizer = torch.optim.Adam([w_frequency_offsets, w_phase_shift_offsets], lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)

    frames = []

    n_iterations = 700

    for idx in range(opt.n_input_views):
        save_image(gt_images[idx], f"{opt.output_dir}/gt{idx}.jpg", normalize=True)

    for i in range(n_iterations):
        noise_w_frequencies = 0.03 * torch.randn_like(w_frequencies) * (n_iterations - i) / n_iterations
        noise_w_phase_shifts = 0.03 * torch.randn_like(w_phase_shifts) * (n_iterations - i) / n_iterations
        with torch.cuda.amp.autocast():
            all_frames = []
            for idx in range(opt.n_input_views):
                frame, _ = generator.forward_with_frequencies(
                    w_frequencies + noise_w_frequencies + w_frequency_offsets,
                    w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets,
                    h_mean=gt_h_means[idx],
                    v_mean=gt_v_means[idx],
                    lock_view_dependence=lock_view_dependence and opt.use_view_lock_for_optimization,
                    **options
                )
                all_frames.append(frame)

        all_frames = torch.cat(all_frames, dim=0)
        # loss = torch.nn.L1Loss()(all_frames, gt_images)
        loss = torch.nn.MSELoss()(all_frames, gt_images)
        loss = loss.mean()

        print(f"{i + 1}/{n_iterations}: {loss.item()} {scheduler.get_lr()}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 25 == 0:
            save_image(all_frames[0], f"{opt.output_dir}/{i}.jpg", normalize=True)

        del frame
        del all_frames

        # Every 25 rounds make images of all possible gt_poses
        if i % 25 == 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for idx, (h, v) in enumerate(zip(image_h_means, image_v_means)):
                        img, _ = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets,
                                                                           w_phase_shifts + w_phase_shift_offsets,
                                                                           h_mean=h,
                                                                           v_mean=v,
                                                                           max_batch_size=opt.max_batch_size,
                                                                           lock_view_dependence=lock_view_dependence,
                                                                           **render_options)
                        save_image(img, f"{opt.output_dir}/{i}_{idx}.jpg", normalize=True)

        # Every round add the first gt pose to the frames
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                img, _ = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets,
                                                                   w_phase_shifts + w_phase_shift_offsets,
                                                                   h_mean=image_h_means[0],
                                                                   v_mean= image_v_means[0],
                                                                   max_batch_size=opt.max_batch_size,
                                                                   lock_view_dependence=lock_view_dependence,
                                                                   **render_options)
                frames.append(tensor_to_PIL(img))

        scheduler.step()

    # Generate a view trajectories around the final result
    additional_frames = generate_additional_frames(
        generator=generator,
        render_options=render_options,
        num_frames=opt.num_frames,
        frequencies=w_frequencies + w_frequency_offsets,
        phase_shifts=w_phase_shifts + w_phase_shift_offsets,
        lock_view_dependence=lock_view_dependence
    )
    frames.extend(additional_frames)

    create_gif(frames, opt.output_dir)
    # create_mp4(frames, opt.output_dir)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('generator_path', type=str)
    parser.add_argument('image_path', type=str)
    parser.add_argument('--output_dir', type=str, default='inverse_images')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=128)
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--n_input_views', type=int, default=1)
    parser.add_argument('--use_view_lock_for_optimization', action='store_true')

    opt = parser.parse_args()
    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    main(opt, device)


# python inverse_render.py ../models/shapenetships_sym_loss_hierarchical_72900/ /home/gitaar9/AI/TNO/shapenet_renderer/ship_renders_train_upper_hemisphere_30_fov/1a2b1863733c2ca65e26ee427f1e5a4c/rgb/000015.png --num_frames=30 --max_batch_size=100000
