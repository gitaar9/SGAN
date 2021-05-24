import os

import torch
from PIL import Image
from skimage import metrics
from torchvision import transforms


def load_image_from_paths(img_paths, img_size=128):
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256),
         transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    images = []
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0)
        images.append(img)
    return torch.cat(images, dim=0).permute(0, 2, 3, 1).contiguous().numpy()


def load_generated_images(folder, last_epoch, amount_of_images=4, img_size=128):
    img_paths = [os.path.join(folder, f"{last_epoch}_{i}.jpg") for i in range(amount_of_images)]
    return load_image_from_paths(img_paths, img_size)


def load_gt_images(folder, amount_of_images=4, img_size=128):
    img_paths = [os.path.join(folder, f"00000{i}.png") for i in range(amount_of_images)]
    return load_image_from_paths(img_paths, img_size)


def calculate_ssim_for_folder(generated_folder, gt_folder, amount_of_images = 4, last_epoch = 675, img_size = 128):
    generated_images = load_generated_images(generated_folder, last_epoch, amount_of_images, img_size)
    gt_images = load_gt_images(gt_folder, amount_of_images, img_size)

    ssims = []
    for image_idx in range(gt_images.shape[0]):
        ssim = metrics.structural_similarity(
            generated_images[image_idx],
            gt_images[image_idx],
            multichannel=True,
            data_range=2,
        )
        ssims.append(ssim)
    return ssims


def main():
    last_epoch = 675
    amount_of_images = 4
    img_size = 128

    car_ids = ['1b1a7af332f8f154487edd538b3d83f6', '1c3c8952b92d567e61c6c61410fc904b',
               '1cb95c00d3bf6a3a58dbdf2b5c6acfca', '1f7393970917e558b4a20251cec15600']

    for car_id in car_ids:
        gt_folder = f'/samsung_hdd/Files/AI/TNO/S-GAN-prerelease/S-GAN-real_prerelease/inverse_images/cars/{car_id}/rgb'
        # result_folder = f'/samsung_hdd/Files/AI/TNO/S-GAN-prerelease/S-GAN-real_prerelease/inverse_images/l1/white_back/cars_inference/{car_id}'
        result_folder = f'/samsung_hdd/Files/AI/TNO/S-GAN-prerelease/S-GAN-real_prerelease/inverse_images/l1/white_back/no_mirror/cars_inference/{car_id}'
        ssims = calculate_ssim_for_folder(result_folder, gt_folder, amount_of_images, last_epoch, img_size)

        print(f"Sims for {car_id}: {ssims}|| Mean: {(ssims[2] + ssims[3]) / 2}")


if __name__ == '__main__':
    main()

