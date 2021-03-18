import argparse
import math
import os

from torchvision.utils import save_image

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, default='')
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--num_images', type=int, default=10)
parser.add_argument('--resolution', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--multi_view', action='store_true')
parser.add_argument('--num_steps', type=int, default=-1)
parser.add_argument('--debug', action='store_true')
opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok=True)

with torch.no_grad():
  generator = torch.load(opt.path, map_location=torch.device(device))
  generator.subsample_keep_percentage = -1
  generator.eval()

  if opt.multi_view:
    generator.update_points(opt.resolution if opt.resolution != None else generator.img_size, opt.batch_size, opt.num_steps if opt.num_steps != -1 else generator.num_steps)
    fake_imgs = []
#     z = torch.rand((opt.batch_size, generator.z_dim), device=device) * 2 - 0.5
    z = torch.randn((opt.batch_size, generator.z_dim), device=device)

    for v in range(5):
      for h in range(5):
        fake_imgs.append(generator(z, h_stddev=0, v_stddev=0, h_mean=(h-2)/5. + math.pi*0.5 + 0.001, v_mean=-(v-2)/5 + math.pi*0.5).squeeze())
        
    fake_imgs = torch.stack(fake_imgs)
    save_image(fake_imgs.data, os.path.join(opt.output_dir, 'multi-view.png'), nrow=5, normalize=True)
  elif opt.debug:
    generator.update_resolution(opt.resolution if opt.resolution != None else generator.resolution, 1, generator.num_steps)
    fake_imgs = []
#     z = torch.rand((opt.batch_size, generator.z_dim), device=device) * 2 - 0.5
    z = torch.randn((opt.batch_size, generator.z_dim), device=device)

    for i in range(1):
        fake_imgs.append(generator(z, h_stddev=0, v_stddev=0, h_mean=math.pi*0.5 - 0.001, v_mean=math.pi*0.5).squeeze())
        fake_imgs.append(generator(z, h_stddev=0, v_stddev=0, h_mean=math.pi*0.5, v_mean=math.pi*0.5).squeeze())
        fake_imgs.append(generator(z, h_stddev=0, v_stddev=0, h_mean=math.pi*0.5 + 0.001, v_mean=math.pi*0.5).squeeze())
    fake_imgs = torch.stack(fake_imgs)
    save_image(fake_imgs.data, os.path.join(opt.output_dir, 'multi-view.png'), nrow=3, normalize=True)
  else:
    generator.update_resolution(opt.resolution if opt.resolution != None else generator.resolution, opt.batch_size, generator.num_steps if has_attr(generator, 'num_steps') else 1)#generator.num_steps)

    for i in range(opt.num_images):
#       z = torch.rand((opt.batch_size, generator.z_dim), device=device) * 2 - 0.5
      z = torch.randn((opt.batch_size, generator.z_dim), device=device)
      fake_img = generator(z)
      save_image(fake_img, os.path.join(opt.output_dir, f'{i}.png'), normalize=True)