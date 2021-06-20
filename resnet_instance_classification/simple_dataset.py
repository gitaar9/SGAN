import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset


class SimpleShapenetRecognitionDataset(Dataset):
    """SimpleShapenetRecognitionDataset"""

    def __init__(self, root_dir, extra_root_dir=None, start_idx=0, amount_images=30, amount_extra_images=30,
                 transform=None):
        """
        :param root_dir: The root directory of the ShapeNetCars dataset
        :param transform: The transform that should be applied to every image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.load_data(
            root_dir,
            extra_root_dir=extra_root_dir,
            start_idx=start_idx,
            amount_images=amount_images,
            amount_extra_images=amount_extra_images
        )

    def load_data(self, root_dir, extra_root_dir=None, start_idx=0, amount_images=30, amount_extra_images=0):
        paths = self.load_img_by_idx(root_dir, range(start_idx, start_idx + amount_images), 'rgb')
        if extra_root_dir is not None and amount_extra_images > 0:
            paths.extend(self.load_img_by_idx(
                extra_root_dir,
                range(amount_images, amount_images + amount_extra_images)
            ))
        return paths

    def load_img_by_idx(self, folder, idx_range, extra_folder=None):
        object_folders = self.retrieve_object_folders(folder)

        paths = []
        for class_label, object_folder in enumerate(object_folders):
            if extra_folder is not None:
                object_folder = os.path.join(object_folder, extra_folder)
            for image_idx in idx_range:
                paths.append((os.path.join(object_folder, f"{image_idx:06}.png"), class_label))
        return paths

    @staticmethod
    def retrieve_object_folders(root_dir):
        object_folders = glob.glob(os.path.join(root_dir, '*'))
        object_folders = [f for f in object_folders if '.' not in f]
        object_folders = list(sorted(object_folders))
        print(f"Found {len(object_folders)} objects")
        return object_folders

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, class_label = self.data[idx]
        image = cv2.imread(image_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(f'Problem with {image_path}')
            exit()

        if image is None:
            print(f"Wrong path: {image_path}")
        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, class_label

    def show_image(self, idx):
        img, label = self[idx]
        print(label)
        if isinstance(img, torch.Tensor):
            img = self.tensor_to_PIL(img)
        img.show()

    @staticmethod
    def tensor_to_PIL(img):
        img = img.squeeze() * 0.5 + 0.5
        return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
