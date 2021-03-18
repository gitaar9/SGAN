import argparse
import torch
import math
import glob
import numpy as np
import sys
import os
from tqdm import tqdm
from torchvision.utils import save_image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()
    
def generate_img(generator, z, **kwargs):
    
    with torch.no_grad():
        img, depth_map = generator.staged_forward(z, **kwargs)
        tensor_img = img.detach()
        
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img, depth_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    opt = parser.parse_args()
    
    os.makedirs(opt.output_dir, exist_ok=True)

    generator = torch.load(opt.path, map_location=torch.device(device))
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    options_dict = {
        'num_steps':36,
        'img_size':opt.image_size,
        'hierarchical_sample':True,
        'psi':0.5,
        'ray_start':0.88,
        'ray_end':1.12,
        'v_stddev': 0,
        'h_stddev': 0,
        'h_mean': 0 + math.pi/2,
        'v_mean': 0 + math.pi/2,
        'fov': 12,
        'lock_view_dependence': opt.lock_view_dependence,
        'white_back':False,
        'last_back': True,
        'clamp_mode': 'relu',
        'nerf_noise': 0,
    }
    
    face_angles = [-0.5, -0.25, 0., 0.25, 0.5]

    face_angles = [a + options_dict['h_mean'] for a in face_angles]

    for seed in tqdm(opt.seeds):
        images = []
        for i, yaw in enumerate(face_angles):
            options_dict['h_mean'] = yaw
            torch.manual_seed(seed)
            z = torch.randn((1, 256), device=device)
            img, tensor_img, depth_map = generate_img(generator, z, **options_dict)
            save_image(tensor_img, os.path.join(opt.output_dir, f"img_{seed}_{yaw}_.png"), normalize=True)
            images.append(tensor_img)
        save_image(torch.cat(images), os.path.join(opt.output_dir, f'grid_{seed}.png'), normalize=True)
