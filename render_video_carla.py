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
parser.add_argument('--output_dir', type=str, default='vids')
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


def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


generator = torch.load(opt.path, map_location=torch.device(device))
ema_file = opt.path.split('generator')[0] + 'ema.pth'
ema = torch.load(ema_file)
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()

trajectory = []
for t in np.linspace(0, 1, curriculum['num_frames']):
    # pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2  # these dont work
    # yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
    # pitch = math.pi/2 * (1 - t)
    # yaw = 2 * math.pi * t
    pitch = ((math.pi / 2 * 85 / 90) * t)
    yaw = (2 * math.pi * t) - curriculum['h_mean']
    fov = 30
        
    trajectory.append((pitch, yaw, fov))

frame_repeat = 5

for seed in opt.seeds:
    frames = []
    depths = []
    output_name = f'{seed}.mp4'
    writer = skvideo.io.FFmpegWriter(os.path.join(opt.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})

    torch.manual_seed(seed)
    z = torch.randn(1, 256, device=device)

    with torch.no_grad():
        for pitch, yaw, fov in tqdm(trajectory):
            curriculum['h_mean'] = yaw #  + 3.14/2
            curriculum['v_mean'] = pitch #  + 3.14/2
            curriculum['fov'] = fov
            curriculum['h_stddev'] = 0
            curriculum['v_stddev'] = 0

            frame, depth_map = generator.staged_forward(z, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
            frames.append(tensor_to_PIL(frame))

        for frame in frames:
            for _ in range(frame_repeat):
                writer.writeFrame(np.array(frame))

        writer.close()
