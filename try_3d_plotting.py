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
from generators.volumetric_rendering import get_initial_rays_trig, transform_sampled_points, fancy_integration
from siren import siren
import fid_evaluation

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy

from torch_ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
from PIL import Image


def fancy_plot(generator, z, device, points, camera_origin):
    ax = plt.axes(projection='3d')
    query_uniformly_sampled_points(generator, z, device, ax)

    if not isinstance(points, np.ndarray):
        points = points.cpu().detach().numpy().squeeze()
    if not isinstance(camera_origin, np.ndarray):
        camera_origin = camera_origin.cpu().detach().numpy().squeeze()
    plot_3d_points(points, max_points=7500, extra_point=camera_origin, plt_ax=ax, alpha=.01, strange_alpha=True)


def fancy_plotting(generator, z, device, metadata):
    ax = plt.axes(projection='3d')
    # Plot the object roughly
    query_uniformly_sampled_points(generator, z, device, ax)

    # Plot the rays and camera position
    transformed_points, transformed_ray_directions_expanded, camera_origin, z_vals = sample_points_as_in_generator(**metadata, device=device)
    transformed_points_numpy = transformed_points.cpu().detach().numpy().squeeze()
    camera_origin = camera_origin.cpu().detach().numpy().squeeze()
    plot_3d_points(transformed_points_numpy, max_points=5000, extra_point=camera_origin, plt_ax=ax, alpha=.005)

    # Create the img
    course_output = query_siren_as_in_generator(generator, transformed_points,
                                                transformed_ray_directions_expanded, z, device, **metadata)
    show_siren_output_as_image(course_output, z_vals, device, **metadata)

    limit = 1.
    ax.set_xlim3d(-limit, limit)
    ax.set_ylim3d(-limit, limit)
    ax.set_zlim3d(-limit, limit)

    # Show the plot after the image is created
    plt.show()


def invert_point_tensor(point_tensor):
    # Old implementation
    # point_tensor_numpy = point_tensor.cpu().detach().numpy().squeeze()
    # inverse_point_tensor_numpy = point_tensor_numpy.copy()
    # old_x = inverse_point_tensor_numpy[:, 0].copy()
    # inverse_point_tensor_numpy[:, 0] = inverse_point_tensor_numpy[:, 2].copy()
    # inverse_point_tensor_numpy[:, 2] = old_x
    # inverse_point_tensor = torch.from_numpy(inverse_point_tensor_numpy).unsqueeze(0).float().to(device)

    # Swaps x and z axis
    return point_tensor[:, :, [2, 1, 0]]


def mirror_experiment(generator, z, device, **metadata):
    transformed_points, transformed_ray_directions_expanded, camera_origin, z_vals = sample_points_as_in_generator(**metadata, device=device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # Normal render
            course_output = query_siren_as_in_generator(generator, transformed_points, transformed_ray_directions_expanded, z, device, **metadata)
            show_siren_output_as_image(course_output, z_vals, device, **metadata)

            # Inverse render
            # Switch x and z axis for points
            inverse_transformed_points = invert_point_tensor(transformed_points)
            # Switch x and z axis for directions
            inverse_transformed_ray_directions_expanded = invert_point_tensor(transformed_ray_directions_expanded)
            course_output = query_siren_as_in_generator(generator, inverse_transformed_points, inverse_transformed_ray_directions_expanded, z, device, **metadata)
            show_siren_output_as_image(course_output, z_vals, device, tag=True, **metadata)

    # Show original points and camera position
    # yaw = math.atan2(camera_origin[2], camera_origin[0])
    yaw = torch.atan2(camera_origin[:, 2], camera_origin[:, 0])
    inverse_yaw = torch.atan2(camera_origin[:, 0], camera_origin[:, 2])
    yaw = yaw.cpu().detach().numpy().squeeze()
    inverse_yaw = inverse_yaw.cpu().detach().numpy().squeeze()
    print("Original yaw: ", math.degrees(yaw), yaw)
    print("Inverse yaw: ", math.degrees(inverse_yaw), inverse_yaw)
    plt.figure(1, (8, 8))
    transformed_points_numpy = transformed_points.cpu().detach().numpy().squeeze()
    fancy_plot(generator, z, device, transformed_points_numpy, camera_origin)

    # Show inverted points and camera position
    plt.figure(2, (8, 8))
    camera_origin = camera_origin.cpu().detach().numpy().squeeze()
    inverse_camera_origin = np.asarray([camera_origin[2], camera_origin[1], camera_origin[0]])
    inverse_yaw = math.atan2(inverse_camera_origin[2], inverse_camera_origin[0])
    print("Inverse yaw: ", math.degrees(inverse_yaw), inverse_yaw)
    inverse_transformed_points_numpy = inverse_transformed_points.cpu().detach().numpy().squeeze()
    fancy_plot(generator, z, device, inverse_transformed_points_numpy, inverse_camera_origin)
    plt.show()


def show_siren_output_as_image(siren_output, z_vals, device, img_size, tag=False, **kwargs):
    batch_size = 1
    pixels, depth, weights = fancy_integration(siren_output, z_vals, device=device,
                                               white_back=kwargs.get('white_back', False),
                                               last_back=kwargs.get('last_back', False),
                                               clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

    pixels = pixels.reshape((batch_size, img_size, img_size, 3))
    pixels = pixels.permute(0, 1, 2, 3).contiguous() * 2 - 1
    print(pixels.shape)

    pixels = pixels.cpu().detach().numpy().squeeze()
    pixels = (((pixels + 1) / 2) * 255).astype(np.uint8)
    print(np.min(pixels), np.max(pixels))
    if tag:
        pixels = np.fliplr(pixels)
        pixels[-10:, -10:, :] = [255, 0, 0]
    img = Image.fromarray(pixels, 'RGB')
    img.show()

    print(pixels.shape)


def plot_3d_points(coordinates_list, max_points=15000, extra_point=None, plt_ax=None, strange_alpha=False, alpha=1.):
    if len(coordinates_list) > max_points:
        coordinates_list = coordinates_list.copy()
        idxs = np.random.choice(np.arange(len(coordinates_list)), 15000, replace=False)
        coordinates_list = coordinates_list[idxs]
    if plt_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        ax = plt_ax

    rgba = [(.2, .2, .6, alpha) for _ in range(len(coordinates_list))]
    if strange_alpha:  # make pixels further away from mean less transparant
        c_l_np = np.asarray(coordinates_list)
        avg_coord = np.mean(c_l_np, axis=0)
        avg_coord = np.concatenate([np.expand_dims(avg_coord, 0)] * c_l_np.shape[0], axis=0)
        distance = (c_l_np - avg_coord) ** 2
        distance = np.sum(distance, axis=1)
        distance /= np.max(distance)
        distance /= 2
        distance **= 2
        rgba = [(0.2, 0.2, .6, a) for a in list(distance)]
    ax.scatter(*list(zip(*coordinates_list)), marker='.', c=rgba)
    if extra_point is not None:
        ax.scatter(*extra_point, marker='*')
    if plt_ax is None:
        plt.show()


def sample_points_as_in_generator(img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, device, sample_dist=None, **kwargs):
    batch_size = 1
    points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size),
                                                           device=device, fov=fov, ray_start=ray_start,
                                                           ray_end=ray_end)  # batch_size, pixels, num_steps, 1
    transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw, camera_origin = transform_sampled_points(
        points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean,
        device=device, mode=sample_dist)
    transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)

    transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
    transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
    transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size,
                                                                                      img_size * img_size * num_steps,
                                                                                      3)
    print("Yaw: ", math.degrees(yaw[0][0]), " Pitch: ", math.degrees(pitch[0][0]))
    return transformed_points, transformed_ray_directions_expanded, camera_origin, z_vals


def query_siren_as_in_generator(generator, transformed_points, transformed_ray_directions_expanded, z, device, img_size, num_steps, max_batch_size=20000, **kwargs):
    truncated_frequencies, truncated_phase_shifts = siren_freqeuncies_and_phase_shifts(generator, z)
    batch_size = 1
    coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=device)

    for b in range(batch_size):
        head = 0
        while head < transformed_points.shape[1]:
            tail = head + max_batch_size
            coarse_output[b:b + 1, head:tail] = generator.siren.forward_with_frequencies_phase_shifts(
                transformed_points[b:b + 1, head:tail], truncated_frequencies[b:b + 1], truncated_phase_shifts[b:b + 1],
                ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
            head += max_batch_size

    return coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)


def siren_freqeuncies_and_phase_shifts(generator, z, psi=.7):
    # Get the activation function settings from the mapping network
    generator.generate_avg_frequencies()
    raw_frequencies, raw_phase_shifts = generator.siren.mapping_network(z)

    truncated_frequencies = generator.avg_frequencies + psi * (raw_frequencies - generator.avg_frequencies)
    truncated_phase_shifts = generator.avg_phase_shifts + psi * (raw_phase_shifts - generator.avg_phase_shifts)
    return truncated_frequencies, truncated_phase_shifts


def uniformly_sample_3d_points(sample_per_axis, show_points=False):
    import itertools
    max = .25
    y_max = .25
    x_ = np.linspace(-max, max, sample_per_axis)
    y_ = np.linspace(-y_max, y_max, sample_per_axis)
    z_ = np.linspace(-max, max, sample_per_axis)

    # x_ = np.linspace(-2., 2., sample_per_axis)
    # y_ = np.linspace(-3., 3., sample_per_axis)
    # z_ = np.linspace(-1., 3., sample_per_axis)

    coordinates_list = list(itertools.product(x_, y_, z_))

    if show_points:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*list(zip(*coordinates_list)), marker='o')
        plt.show()

    return coordinates_list


def plot_points(coordinates_list, rgb, alpha, plt_ax=None):
    # need to plot the points using their alpha/rgb
    rgb = rgb.cpu().detach().numpy().squeeze()
    alpha = alpha.cpu().detach().numpy().squeeze()
    print("RGB array shape: ", rgb.shape, " Min and max: ", np.min(rgb), np.max(rgb))
    print("Alpha array shape: ", alpha.shape, " Min and max: ", np.min(alpha), np.max(alpha))
    rgba = np.concatenate((rgb, np.expand_dims(alpha, axis=1)), axis=1) # Combine rgb and alpha

    # Filter out points with to little alpha or completely white points
    zipped = [(c, (r, g, b, a)) for c, (r, g, b, a) in list(zip(coordinates_list, rgba)) if a > 0.01 and ((b + g + r) < 2.9)]
    coordinates_list, rgba = zip(*zipped)
    print("Nr of points left after filtering: ", len(coordinates_list))

    # Plot the points in 3D
    if plt_ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    else:
        ax = plt_ax
    ax.scatter(*list(zip(*coordinates_list)), marker='.', c=list(rgba))

    if plt_ax is None:
        limit = .25
        ax.set_xlim3d(-limit, limit)
        ax.set_ylim3d(-limit, limit)
        ax.set_zlim3d(-limit, limit)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if plt_ax is None:
        plt.show()


def query_uniformly_sampled_points(generator, z, device, ax=None):
    print("Generating points:")
    # Get the activation function settings from the mapping network
    truncated_frequencies, truncated_phase_shifts = siren_freqeuncies_and_phase_shifts(generator, z)

    coordinates_list = uniformly_sample_3d_points(55)
    transformed_points = torch.from_numpy(np.asarray([coordinates_list])).float().to(device)
    ray_directions = torch.from_numpy(np.asarray([[[1, 0, 1]] * len(coordinates_list)])).float().to(device)
    print(transformed_points.shape)
    print(ray_directions.shape)

    course_output = generator.siren.forward_with_frequencies_phase_shifts(
        transformed_points,
        truncated_frequencies,
        truncated_phase_shifts,
        ray_directions=ray_directions
    )

    rgbs = course_output[..., :3]
    sigmas = course_output[..., 3:]
    alphas = 1 - torch.exp(-1 * (F.relu(sigmas)))

    print(rgbs.shape)
    print(alphas.shape)

    plot_points(coordinates_list, rgbs, alphas, ax)
    torch.cuda.empty_cache()


def normal_render(generator, z, device, metadata):
    print("Generating images:")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
            copied_metadata['img_size'] = 256
            gen_imgs = generator.staged_forward(z.to(device),  **copied_metadata)[0]  # Can also be fixed z
    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"test_output.png"), nrow=5, normalize=True)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
            copied_metadata['h_mean'] += math.pi / 2
            copied_metadata['img_size'] = 256
            gen_imgs = generator.staged_forward(z.to(device),  **copied_metadata)[0]
    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"test_output_tilted.png"), nrow=5, normalize=True)
    del gen_imgs
    torch.cuda.empty_cache()


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def train(opt):
    device = torch.device(0)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z = z_sampler((25, 256), device='cpu', dist=metadata['z_dist'])

    SIREN = getattr(siren, metadata['model'])
    scaler = torch.cuda.amp.GradScaler()

    if opt.load_dir != '':
        generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
        discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)
    else:
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
        discriminator = getattr(discriminators, metadata['discriminator'])().to(device)

    generator.set_device(device)

    # Dont know whether this is needed
    if scaler.get_scale() < 1:
        scaler.update(1.)

    metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), metadata['grad_clip'])

    # Generate some images
    N_images = 1
    z = z_sampler((N_images, metadata['latent_dim']), device=device, dist=metadata['z_dist'])

    metadata['img_size'] = 256
    metadata['max_batch_size'] = 20000
    del metadata['generator']

    # Experiments for symmetrical loss
    mirror_experiment(generator, z, device, **metadata)

    exit()
    # Experiments for symmetrical loss
    mirror_experiment(generator, z, device, **metadata)
    # Standard test rendering
    normal_render(generator, z, device, metadata)
    # Query uniformly sampled points to siren:
    query_uniformly_sampled_points(generator, z, device)
    # Fancy plotting
    fancy_plotting(generator, z, device, metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)

    train(opt)
