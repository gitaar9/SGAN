import glob
import os
from functools import cmp_to_key

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import metrics
from torchvision import transforms


def glue_images_from_path(paths, image_size=128):
    images = []
    for p in paths:
        image = cv2.imread(p)  # Load the image
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        images.append(image)
    final_image = np.hstack(images)
    return final_image


def create_gt_generated_image_samples(generated_image_paths, gt_image_paths, image_size=128):
    image = np.vstack(
        [glue_images_from_path(generated_image_paths, image_size), glue_images_from_path(gt_image_paths, image_size)]
    )
    return image


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


def cmp_paths(a, b):
    l = [
        '1b1a7af332f8f154487edd538b3d83f6',
        '4ac021653126a29e98618a1ba17f086b',
        '59e01fab310fa8c49c9f1f5abaab90a7',
        'a3e7603c0d9ef56280e74058ee862f05',
        '39d161909e94d99e61b9ff60b1be412',
        '3d358a98f90e8b4d5b1edf5d4f643136',
        '5c5908b7a19d8df7402ac33e676077b1',
        '3ae62dd54a2dd59ad7d2e5d7d40456b9',
        '999007a25b5f3db3d92073757fe1175e',
        '70b8730b003c345159139efcde1fedcb',
        '99f49d11dad8ee25e517b5f5894c76d9',
        '93bb1cd910f054818c2e7159929c406f',
        'ee0edec0ac0578082ea8e4c752a397ac',
        '625861912ac0d62651a95aaa6caba1d3',
        '99fce87e4e80e053374462542bf2aa29',
        'afa0436d9cb1b19ec8c241cb24f7e0ac',
        'dcfdb673231b64ea413908c0e169330',
        'af08f280395cd31719f6a4bcc93433',
        '77a759df0166630adeb0f0d7312576e9',
        '1cc85c2c1580088aad48f475ce080a1f',
        'f6a93b95e10a8b2d6aea15d30373dbc0',
        '2c3a3033d248d05851a95aaa6caba1d3',
        'd0bf09ffa93ba912582e5e3e7771ea25',
        'bf52cda8de7e5eff36dfef0450f0ee37',
        '8b7b6c2a5c664ca6efe5f291bc2f5fd0'
    ]
    a = a.split('/')[-1]
    b = b.split('/')[-1]
    if a in l:
        a = l.index(a)
    else:
        a = 30
    if b in l:
        b = l.index(b)
    else:
        b = 30
    if a > b:
        return 1
    elif a == b:
        return 0
    else:
        return -1

def main():

    output_folders = [
        # '/samsung_hdd/Files/AI/TNO/remote_folders/train_pose_from_test_image_remotes/car_view_synthesis_test_set_output/car_view_synthesis_test_set_output',
        # '/samsung_hdd/Files/AI/TNO/remote_folders/train_pose_from_test_image_remotes/car_view_synthesis_test_set_output_350/car_view_synthesis_test_set_output_350',
        '/samsung_hdd/Files/AI/TNO/remote_folders/train_pose_from_test_image_remotes/car_view_synthesis_test_set_output_no_view_lock/car_view_synthesis_test_set_output_no_view_lock',
        '/samsung_hdd/Files/AI/TNO/remote_folders/train_pose_from_test_image_remotes/car_view_synthesis_test_set_output_no_view_lock_at_all/car_view_synthesis_test_set_output_no_view_lock_at_all',
        '/samsung_hdd/Files/AI/TNO/remote_folders/train_pose_from_test_image_remotes/car_view_synthesis_test_set_output_no_view_lock_at_all_700/car_view_synthesis_test_set_output_no_view_lock_at_all_700',
        '/samsung_hdd/Files/AI/TNO/remote_folders/train_pose_from_test_image_remotes/car_view_synthesis_test_set_output_no_view_lock_700/car_view_synthesis_test_set_output_no_view_lock_700',
    ]
    reference_folder = '/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_view_synthesis_test_set'
    img_size = 128

    samples = []
    for output_folder in output_folders:
        object_folders = glob.glob(os.path.join(output_folder, '*'))
        object_folders.sort(key=cmp_to_key(cmp_paths))

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
        # print(ssims)
        print(f"Average SSIM over {len(generated_images)} images: {np.mean(ssims):.3f}:{np.std(ssims):.3f}\t(median: {np.median(ssims):.3f})")

        samples.append(glue_images_from_path(generated_images_paths))
        #

    samples.append(glue_images_from_path(ground_truth_image_paths))
    input_image_paths = [os.path.join('/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_view_synthesis_test_set', o_id, 'rgb', '000000.png') for o_id in objects_ids]
    samples.append(glue_images_from_path(input_image_paths))

    # You may need to convert the color.
    min_width = min([s.shape[1] for s in samples])
    image = np.vstack([s[:, :min_width] for s in samples])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image)
    im_pil.show()
    # cv2.imshow('image', image.astype(np.uint8))  # Show the image
    # cv2.waitKey(0)
    print(objects_ids)

if __name__ == '__main__':
    main()

