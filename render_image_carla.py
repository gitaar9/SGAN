import argparse
import math
import os

from torchvision.utils import save_image

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import skvideo.io
import curriculums


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
parser.add_argument('--output_dir', type=str, default='images')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_batch_size', type=int, default=2400000)
parser.add_argument('--depth_map', action='store_true')
parser.add_argument('--lock_view_dependence', action='store_true')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--num_frames', type=int, default=36)
opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok=True)

curriculum = {
    'num_steps': 30,
    'img_size': opt.image_size,
    'hierarchical_sample': True,
    'psi': 0.7,
    'ray_start': 0.75,
    'ray_end': 1.25,
    'v_stddev': 0,
    'h_stddev': 0,
    'h_mean': math.pi * 0.5,
    'v_mean': math.pi / 4 * 85 / 90,
    'fov': 30,
    'lock_view_dependence': opt.lock_view_dependence,
    'white_back': True,
    'last_back': False,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
    'num_frames': opt.num_frames,
}

curriculum['v_mean'] += ((math.pi / 4 * 85 / 90) / 10) * 7
curriculum['h_mean'] += (math.pi / 100) * 5

print(f"v_mean: {curriculum['v_mean']}, h_mean: {curriculum['h_mean']}")


def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


generator = torch.load(opt.path, map_location=torch.device(device))
ema_file = opt.path.split('generator')[0] + 'ema.pth'
ema = torch.load(ema_file)
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()


for seed in opt.seeds:
    frames = []
    depths = []
    output_name = f'{seed}.mp4'
    writer = skvideo.io.FFmpegWriter(os.path.join(opt.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})

    torch.manual_seed(seed)
    z = torch.randn(1, 256, device=device)

    with torch.no_grad():
        frame, depth_map = generator.staged_forward(z, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
    frame = tensor_to_PIL(frame)
    frame.show()
    # Image.fromarray(frame, 'RGB').show()
    # print(frame.shape)