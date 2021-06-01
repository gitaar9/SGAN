import glob
import os

import numpy as np
import torch
from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset


class ShapeNetCarsRecognitionDataset(Dataset):
    """ShapeNetCars dataset for a recognition task"""

    def __init__(self, root_dir, is_train=True, amount_of_images_per_object=31, transform=None):
        """
        :param root_dir: The root directory of the ShapeNetCars dataset
        :param is_train: When this is true only training images are loaded
        :param transform: The transform that should be applied to every image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.train_data = self.load_data(amount_of_images_per_object)

    def load_data(self, amount_of_images_per_object):
        # Read in annotations
        object_folders = glob.glob(os.path.join(self.root_dir, '*'))
        print(f"Found {len(object_folders)} objects")

        train_percentage = .8
        split_idx = int(amount_of_images_per_object * train_percentage)

        train_data = []
        for class_label, object_folder in enumerate(object_folders):
            rnge = range(split_idx) if self.is_train else range(split_idx, amount_of_images_per_object)
            for image_idx in rnge:
                train_data.append((os.path.join(object_folder, f"rgb/{image_idx:06}.png"), class_label))
        return train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        image_path, class_label = self.train_data[idx]
        image = cv2.imread(image_path)

        if image is None:
            print(f"Wrong path: {image_path}")
        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, class_label

    def show_image(self, idx):
        img, _ = self[idx]
        if isinstance(img, torch.Tensor):
            img = self.tensor_to_PIL(img)
        img.show()

    @staticmethod
    def tensor_to_PIL(img):
        img = img.squeeze() * 0.5 + 0.5
        return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


class ShapeNetCarsRecognitionDatasetOnlyLastImageIsTest(ShapeNetCarsRecognitionDataset):
    """ShapeNetCars dataset for a recognition task"""

    def load_data(self, amount_of_images_per_object):
        # Read in annotations
        object_folders = glob.glob(os.path.join(self.root_dir, '*'))
        print(f"Found {len(object_folders)} objects")

        train_data = []
        for class_label, object_folder in enumerate(object_folders):
            if self.is_train:
                for image_idx in range(amount_of_images_per_object - 1):
                    train_data.append((os.path.join(object_folder, f"rgb/{image_idx:06}.png"), class_label))
            else:
                train_data.append((os.path.join(object_folder, f"rgb/{amount_of_images_per_object - 1:06}.png"), class_label))
        return train_data


# root_dir = '/samsung_hdd/Files/AI/TNO/shapenet_renderer/car_renders_train_upper_hemisphere_30_fov_pixel_nerf/cars_val'