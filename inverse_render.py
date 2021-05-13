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
parser.add_argument('--output_dir', type=str, default='inverse_images')
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

lock_view_dependence = False
image_h = 1.7549115333974483  # Speedboat 5
image_v = 0.7326789041951802  # Speedboat 5
image_h = 3.1917140533974484  # halfcontainer 9
image_v = 0.8804224641951802  # halfcontainer 9
image_h = -0.7012666066025517  # sailship 6
image_v = 1.0182758041951803  # sailship 6


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
    'sample_dist': None,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
}

render_options = {
    'img_size': 128,
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

save_image(gt_image, f"{opt.output_dir}/gt.jpg", normalize=True)

for i in range(n_iterations):
    noise_w_frequencies = 0.03 * torch.randn_like(w_frequencies) * (n_iterations - i)/n_iterations
    noise_w_phase_shifts = 0.03 * torch.randn_like(w_phase_shifts) * (n_iterations - i)/n_iterations
    frame, _ = generator.forward_with_frequencies(w_frequencies + noise_w_frequencies + w_frequency_offsets, w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets, **options)
    loss = torch.nn.MSELoss()(frame, gt_image)
    loss = loss.mean()
    
    print(f"{i + 1}/{n_iterations}: {loss.item()} {scheduler.get_lr()}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if i % 1 == 0:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                img, _ = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, h_mean=image_h, max_batch_size=opt.max_batch_size, lock_view_dependence=lock_view_dependence, **render_options)
                frames.append(tensor_to_PIL(img))
                # frames.append(img[0].cpu().permute(1,2,0) * 0.5 + 0.5)
    
    scheduler.step()

    if i % 25 == 0:
        save_image(frame, f"{opt.output_dir}/{i}.jpg", normalize=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for angle in [-0.3, 0, 0.3]:
                    img, _ = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, h_mean=(image_h + angle), max_batch_size=opt.max_batch_size, lock_view_dependence=lock_view_dependence, **render_options)
                    save_image(img, f"{opt.output_dir}/{i}_{angle}.jpg", normalize=True)

trajectory = [] 
for t in np.linspace(0, 1, opt.num_frames):  # Turn around the boat with shifting pitch
    pitch = ((math.pi / 2 * 85 / 90) * t)
    yaw = (2 * math.pi * t) - (math.pi * 0.5)
    fov = 30
    trajectory.append((pitch, yaw))
for t in np.linspace(0, 1, opt.num_frames):  # Turn around the boat at lowest level pitch
    pitch = (math.pi / 2 * 85 / 90)
    yaw = (2 * math.pi * t) - (math.pi * 0.5)
    fov = 30
    trajectory.append((pitch, yaw))

# output_name = opt.output if opt.output else os.path.splitext(os.path.basename(opt.z))[0] + '.mp4'

# frames = []
# depths = []


with torch.no_grad():
    for pitch, yaw in tqdm(trajectory):
        render_options['h_mean'] = yaw
        render_options['v_mean'] = pitch
        with torch.cuda.amp.autocast():
            frame, depth_map = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, max_batch_size=opt.max_batch_size, lock_view_dependence=lock_view_dependence, **render_options)
            frames.append(tensor_to_PIL(frame))
            # depths.append(tensor_to_PIL(depth_map))


output_name = 'inverse_render.gif'
img, *imgs = frames
img.save(fp=os.path.join(f'{opt.output_dir}', output_name), format='GIF', append_images=imgs,
         save_all=True, duration=45, loop=0, interlace=False)

output_name = 'inverse_render.avi'
writer = skvideo.io.FFmpegWriter(os.path.join(f'{opt.output_dir}', output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})

frame_repeat = 2
for frame in frames:
    for _ in range(frame_repeat):
        writer.writeFrame(np.array(frame))
# for depth in depths:
#     writer.writeFrame(np.array(depth))
writer.close()


#python inverse_render.py ../models/shapenetships_sym_loss_hierarchical_72900/ /home/gitaar9/AI/TNO/shapenet_renderer/ship_renders_train_upper_hemisphere_30_fov/1a2b1863733c2ca65e26ee427f1e5a4c/rgb/000015.png --num_frames=30 --max_batch_size=100000
