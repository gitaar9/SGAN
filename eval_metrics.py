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
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'img_size': 128,
    'num_steps': 12,
    'psi': 1,
    'sample_dist': 'gaussian',
    'latent_dim': 256,
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'nerf_noise': 0,
    'last_back': True
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('generator_file', type=str)
    parser.add_argument('--real_image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='temp')
    parser.add_argument('--num_images', type=int, default=2048)
    parser.add_argument('--gpu_type', type=str, default='8000')
    parser.add_argument('--max_batch_size', type=int, default=94800000)

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    print(metrics_dict)