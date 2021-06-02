import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import metrics
from torchvision import transforms


def show_gt_generated_image_samples(generated_image_paths, gt_image_paths, image_size=128):
    gen_images = []
    gt_images = []
    for gen_img_p, gt_img_p in zip(generated_image_paths, gt_image_paths):
        gen_image = cv2.imread(gen_img_p)  # Load the image
        gen_images.append(gen_image)
        gt_image = cv2.imread(gt_img_p)  # Load the image
        gt_image = cv2.resize(gt_image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        gt_images.append(gt_image)
    image = np.vstack([np.hstack(gen_images), np.hstack(gt_images)])
    cv2.imshow('image', image.astype(np.uint8))  # Show the image
    cv2.waitKey(0)


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


def old_main():
    last_epoch = 675
    amount_of_images = 4
    img_size = 128

    car_ids = ['1b1a7af332f8f154487edd538b3d83f6', '1c3c8952b92d567e61c6c61410fc904b',
               '1cb95c00d3bf6a3a58dbdf2b5c6acfca', '1f7393970917e558b4a20251cec15600']
    car_ids = ['minicooper', 'peugot', 'pickup', 'rapide']
    # car_ids = ['rapide']

    for car_id in car_ids:
        gt_folder = f'/samsung_hdd/Files/AI/TNO/S-GAN-prerelease/S-GAN-real_prerelease/inverse_images/cars/{car_id}/rgb'
        # result_folder = f'/samsung_hdd/Files/AI/TNO/S-GAN-prerelease/S-GAN-real_prerelease/inverse_images/l2/cars_inference/{car_id}'
        result_folder = f'/samsung_hdd/Files/AI/TNO/S-GAN-prerelease/S-GAN-real_prerelease/inverse_images/l2/no_mirror/cars_inference/{car_id}'
        ssims = calculate_ssim_for_folder(result_folder, gt_folder, amount_of_images, last_epoch, img_size)

        # print(f"Sims for {car_id}: {ssims}|| Mean: {sum(ssims[1:]) / len(ssims[1:])}")
        print("Sims for {}: {}|| Mean: {:.3f}".format(car_id, ssims, sum(ssims[1:]) / len(ssims[1:])))


def main():
    output_folder = '/samsung_hdd/Files/AI/TNO/remote_folders/train_pose_from_test_image_remotes/car_view_synthesis_test_set_output/car_view_synthesis_test_set_output'
    output_folder = '/samsung_hdd/Files/AI/TNO/remote_folders/train_pose_from_test_image_remotes/car_view_synthesis_test_set_output_350/car_view_synthesis_test_set_output_350'
    reference_folder = '/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_view_synthesis_test_set'
    img_size = 128

    object_folders = glob.glob(os.path.join(output_folder, '*'))

    generated_images_paths = [os.path.join(p, 'rgb', '0.png') for p in object_folders]
    generated_images = load_image_from_paths(generated_images_paths, img_size)

    objects_ids = [p.split('/')[-1] for p in object_folders]
    ground_truth_image_paths = [os.path.join(reference_folder, o_id, 'rgb', '000001.png') for o_id in objects_ids]
    ground_truth_images = load_image_from_paths(ground_truth_image_paths, img_size)

    ssims = []
    for generated_image, gt_image in zip(generated_images, ground_truth_images):
        ssim = metrics.structural_similarity(
            generated_image,
            gt_image,
            multichannel=True,
            data_range=2,
        )
        ssims.append(ssim)

    ssims = np.asarray(ssims)
    print(ssims)
    print(f"Average SSIM over {len(generated_images)} images: {np.mean(ssims):.3f}:{np.std(ssims):.3f}")

    show_gt_generated_image_samples(generated_images_paths, ground_truth_image_paths[:len(generated_images_paths)])


if __name__ == '__main__':
    main()

