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
from torchvision import transforms

def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('generator_path', type=str)
parser.add_argument('image_path', type=str)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--num_frames', type=int, default=128)
parser.add_argument('--max_batch_size', type=int, default=2400000)


opt = parser.parse_args()

generator = torch.load(opt.generator_path, map_location=torch.device(device))
generator.set_device(device)
generator.eval()


if opt.seed is not None:
    torch.manual_seed(opt.seed)


gt_image = Image.open(opt.image_path).convert('RGB')
transform = transforms.Compose(
                    [transforms.Resize(256), transforms.CenterCrop(256), transforms.Resize((opt.image_size, opt.image_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

gt_image = transform(gt_image).to(device).unsqueeze(0)

image_h = 1.7278759594743862
image_v = 1.2610003845659032
options = {
    'img_size': opt.image_size,
    'fov': 30,
    'ray_start': 0.75,
    'ray_end': 1.25,
    'num_steps': 30,
    'h_stddev': 0,
    'v_stddev': 0,
    'h_mean': image_h,
    'v_mean': image_v,
    'hierarchical_sample': True,
    'sample_dist': 'uniform',
    'clamp_mode': 'relu',
    'nerf_noise': 0,
}

render_options = {
    'img_size': 512,
    'fov': 30,
    'ray_start': 0.75,
    'ray_end': 1.25,
    'num_steps': 30,
    'h_stddev': 0,
    'v_stddev': 0,
    'v_mean': image_v,
    'hierarchical_sample': True,
    'sample_dist': None,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
}

print("generating avg freqs")
z = torch.randn((10000, 256), device=device)
with torch.no_grad():
    frequencies, phase_shifts = generator.siren.mapping_network(z)
w_frequencies = frequencies.mean(0, keepdim=True)
w_phase_shifts = phase_shifts.mean(0, keepdim=True)

w_frequency_offsets = torch.zeros_like(w_frequencies)
w_phase_shift_offsets = torch.zeros_like(w_phase_shifts)

w_frequency_offsets.requires_grad_()
w_phase_shift_offsets.requires_grad_()

optimizer = torch.optim.Adam([w_frequency_offsets, w_phase_shift_offsets], lr=1e-2, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)

frames = []

n_iterations = 700

save_image(gt_image, "debug/gt.jpg", normalize=True)

for i in range(n_iterations):
    noise_w_frequencies = 0.03 * torch.randn_like(w_frequencies) * (n_iterations - i)/n_iterations
    noise_w_phase_shifts = 0.03 * torch.randn_like(w_phase_shifts) * (n_iterations - i)/n_iterations
    frame, _ = generator.forward_with_frequencies(w_frequencies + noise_w_frequencies + w_frequency_offsets, w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets, **options)
    loss = torch.nn.MSELoss()(frame, gt_image)
    loss = loss.mean()
    
    print(i, ": ", loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if i % 1 == 0:
        with torch.no_grad():
            img, _ = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, h_mean=image_h, max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
            frames.append(tensor_to_PIL(img))
            # frames.append(img[0].cpu().permute(1,2,0) * 0.5 + 0.5)
    
    scheduler.step()
    print(scheduler.get_lr())

    if i % 5 == 0:
        save_image(frame, f"debug/{i}.jpg", normalize=True)
        with torch.no_grad():
            for angle in [-0.3, 0, 0.3]:
                img, _ = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, h_mean=(image_h + angle), max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
                save_image(img, f"debug/{i}_{angle}.jpg", normalize=True)

trajectory = [] 
for t in np.linspace(0, 1, 24):
    pitch = 0.2 * t
    yaw = 0
    trajectory.append((pitch, yaw))
for t in np.linspace(0, 1, opt.num_frames):
    pitch = 0.2 * np.cos(t * 2 * math.pi)
    yaw = 0.4 * np.sin(t * 2 * math.pi)
    trajectory.append((pitch, yaw))
        
# output_name = opt.output if opt.output else os.path.splitext(os.path.basename(opt.z))[0] + '.mp4'
output_name = 'inverse_render.mp4'
writer = skvideo.io.FFmpegWriter(os.path.join('debug', output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})

# frames = []
# depths = []


with torch.no_grad():
    for pitch, yaw in tqdm(trajectory):
        render_options['h_mean'] = yaw + 3.14/2
        render_options['v_mean'] = pitch + 3.14/2

        frame, depth_map = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
        frames.append(tensor_to_PIL(frame))
        # depths.append(tensor_to_PIL(depth_map))

for frame in frames:
    writer.writeFrame(np.array(frame))
# for depth in depths:
#     writer.writeFrame(np.array(depth))
writer.close()


#python inverse_render.py ../models/shapenetships_sym_loss_hierarchical_72900/ /home/gitaar9/AI/TNO/shapenet_renderer/ship_renders_train_upper_hemisphere_30_fov/1a2b1863733c2ca65e26ee427f1e5a4c/rgb/000015.png --num_frames=30 --max_batch_size=100000