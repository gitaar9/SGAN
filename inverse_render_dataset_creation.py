import argparse
import glob
import math
import os
from pathlib import Path

import torch
from torchvision.utils import save_image

from inverse_render_final_result import find_latent_z, get_ground_truth_images, generate_image_for_given_z


def main(generator_path, n_iterations, seed, image_size, use_view_lock_for_optimization, change_yaw, dataset_path,
         output_dir, max_batch_size, device):
    generator = torch.load(generator_path, map_location=torch.device(device))
    generator.set_device(device)
    generator.eval()

    if seed is not None:
        torch.manual_seed(seed)
        torch.manual_seed(0)

    lock_view_dependence = False

    img_yaws = [(math.pi / 4) * 3, -math.pi/2]
    img_pitches = [math.radians(30), math.radians(10)]

    image_h_means = [-yaw_from_renderer + (math.pi * 0.75) - (0.5 * math.pi) for yaw_from_renderer in img_yaws]
    if change_yaw:  # For new no mirror generator add this
        image_h_means = [h - (math.pi / 100) * 64 for h in image_h_means]
    image_v_means = [(math.pi / 2 * 85 / 90) - pitch_from_renderer for pitch_from_renderer in img_pitches]

    gt_h_means = image_h_means[:1]
    gt_v_means = image_v_means[:1]

    output_h_mean = image_h_means[1]
    output_v_mean = image_v_means[1]

    options = {
        'img_size': image_size,
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

    object_folders = glob.glob(os.path.join(dataset_path, '*'))

    for idx, object_folder in enumerate(object_folders):
        object_id = object_folder.split('/')[-1]
        output_folder_path = os.path.join(output_dir, object_id, 'rgb')
        output_file_path = os.path.join(output_folder_path, "0.png")
        gt_output_file_path = os.path.join(output_folder_path, "gt.png")
        print(f"{idx + 1}/{len(object_folders)}: Finding z for {object_id}")
        if os.path.isfile(output_file_path) and os.path.isfile(gt_output_file_path):
            print("Skipping since output files already exist.")
            continue

        # Load one or multiple images
        gt_images = get_ground_truth_images(object_folder, 1, image_size, device)

        frames, found_frequencies, found_phase_shifts = find_latent_z(
            generator=generator,
            n_iterations=n_iterations,
            gt_images=gt_images,
            gt_h_means=gt_h_means,
            gt_v_means=gt_v_means,
            output_dir=output_dir,
            options=options,
            render_options=render_options,
            max_batch_size=max_batch_size,
            output_image_h_means=[],
            output_image_v_means=[],
            device=device,
            lock_view_dependence=lock_view_dependence,
            use_view_lock_for_optimization=use_view_lock_for_optimization,
            generate_output=False
        )
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)

        # Output img for classification
        img = generate_image_for_given_z(
            generator=generator,
            frequencies=found_frequencies,
            phase_shifts=found_phase_shifts,
            h_mean=output_h_mean,
            v_mean=output_v_mean,
            max_batch_size=max_batch_size,
            lock_view_dependence=lock_view_dependence,
            render_options=render_options
        )
        save_image(img, output_file_path, normalize=True)
        # Output reference gt image
        save_image(frames[0], gt_output_file_path, normalize=True)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('generator_path', type=str)
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--n_iterations', type=int, default=700)
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--use_view_lock_for_optimization', action='store_true')
    parser.add_argument('--change_yaw', action='store_true')

    input_args = parser.parse_args()
    Path(input_args.output_dir).mkdir(parents=True, exist_ok=True)

    main(
        generator_path=input_args.generator_path,
        n_iterations=input_args.n_iterations,
        seed=input_args.seed,
        image_size=input_args.image_size,
        use_view_lock_for_optimization=input_args.use_view_lock_for_optimization,
        change_yaw=input_args.change_yaw,
        dataset_path=input_args.dataset_path,
        output_dir=input_args.output_dir,
        max_batch_size=input_args.max_batch_size,
        device=device
    )

# python inverse_render_dataset_creation.py decent_sncar_generator.pth ../../shapenet_renderer/car_view_synthesis_validation_set/ --output_dir test_dataset_generation/ --image_size 64 --max_batch_size=100000 --use_view_lock_for_optimization