import glob
import os
import shutil
import torch
import math

from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
import datasets
from tqdm import tqdm
import copy
import argparse
import curriculums
import shutil


curriculum = {
    'fov': 30,
    'ray_start': 0.75,
    'ray_end': 1.25,
    'fade_steps': 10000,
    'sample_dist': 'uniform',
    'h_stddev': math.pi,
    'v_stddev': math.pi / 4 * 85 / 90,
    'h_mean': math.pi * 0.5,
    'v_mean': math.pi / 4 * 85 / 90,
    'latent_dim': 256,
    'white_back': True,
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'psi': .7,
}

curriculum.update({
    'batch_size': 1,
    'num_steps': 64,
    'img_size': 128,
    'batch_split': 1,
    'nerf_noise': 0,
    'hierarchical_sample': False,
})


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1


def setup_evaluation(dataset_name, real_image_dir, target_size=128):
    # Only make real images if they haven't been made yet

    generate_new_real_images = not os.path.exists(real_image_dir) or \
                               len(glob.glob(os.path.join(real_image_dir, '*.jpg'))) < 8000

    if generate_new_real_images:
        os.makedirs(real_image_dir)
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, img_size=target_size)
        print('outputting real images...')
        output_real_images(dataloader, 8000, real_image_dir)
        print('...done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('generator_file', type=str)
    parser.add_argument('--real_image_dir', type=str, default=True)
    parser.add_argument('--output_dir', type=str, default='temp')
    parser.add_argument('--num_images', type=int, default=2048)
    parser.add_argument('--gpu_type', type=str, default='8000')
    parser.add_argument('--max_batch_size', type=int, default=94800000)
    parser.add_argument('--dataset_class', type=str, default='ShapenetCarsTest')

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make real images if they dont exist
    setup_evaluation(opt.dataset_class, opt.real_image_dir, target_size=curriculum['img_size'])

    if os.path.exists(opt.output_dir) and os.path.isdir(opt.output_dir):
        shutil.rmtree(opt.output_dir)
    
    os.makedirs(opt.output_dir, exist_ok=False)

    generator = torch.load(opt.generator_file, map_location=device)
    generator.set_device(device)
    ema_file = opt.generator_file.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.eval()

    for img_counter in tqdm(range(opt.num_images)):
        z = torch.randn(1, 256, device=device)

        with torch.no_grad():
            img = generator.staged_forward(z, max_batch_size=opt.max_batch_size, **curriculum)[0].to(device)
            save_image(img, os.path.join(opt.output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))

    metrics_dict = calculate_metrics(opt.output_dir, opt.real_image_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
    print(opt.generator_file)
    print(metrics_dict)
