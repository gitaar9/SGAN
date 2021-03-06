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
parser.add_argument('--seeds', nargs='+', default=[5,6])
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

# yaw_from_renderer = -0.96951337  # 05
# pitch_from_renderer = 0.75085096   # 05
yaw_from_renderer = 3.0204208   # 04
pitch_from_renderer = 1.12459522   # 04

# yaw_from_renderer = 1.48666477  # 06 sailship
# pitch_from_renderer = 0.46525406   # 06 sailship
# yaw_from_renderer = -2.40631589  # 09 halfcontainer
# pitch_from_renderer = 0.6031074   # 09 halfcontainer
yaw_from_renderer = -0.99701956  # 03 halfcontainer
pitch_from_renderer = 0.32446663   # 03 halfcontainer
# yaw_from_renderer = -1.86428757  # 08 romanrow
# pitch_from_renderer = 0.64206937   # 08 romanrow
yaw_from_renderer = -math.pi/4
pitch_from_renderer = math.radians(35)

# new for testing shapenetcars_no_mirror_v3
yaw_from_renderer = 2.66242184
# yaw_from_renderer = math.pi
pitch_from_renderer = 0.10832326


curriculum['v_mean'] = (math.pi / 2 * 85 / 90) - pitch_from_renderer
curriculum['h_mean'] = -yaw_from_renderer + (math.pi * 0.75) + (0.5 * math.pi) + (3.03)  # The last term is specific for shapenetcars_no_mirror_v3

print(f"Pi-GAN space v_mean: {curriculum['v_mean']}, h_mean: {curriculum['h_mean']}")

print(f"v_mean: {curriculum['v_mean']}, h_mean: {curriculum['h_mean']}")


def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


generator = torch.load(opt.path, map_location=torch.device(device))
# ema_file = opt.path.split('generator')[0] + 'ema.pth'
# ema = torch.load(ema_file)
# ema.copy_to(generator.parameters())
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