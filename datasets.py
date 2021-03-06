import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np


class CelebA(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob('/home/ericryanchan/S-GAN/data/celeba/celeba/img_align_celeba/*.jpg')
        self.transform = transforms.Compose(
                    [transforms.Resize(320), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size, img_size), interpolation=0)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class MyBaseDataset(CelebA):
    def __init__(self, img_size, **kwargs):
        super().__init__(img_size)

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((img_size, img_size), interpolation=0)
            ]
        )

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index]).convert('RGB')
        X = self.transform(X)

        return X, 0


class ShapenetShips(MyBaseDataset):
    def __init__(self, img_size, **kwargs):
        super().__init__(img_size)

        # self.data = glob.glob('/home/gitaar9/AI/TNO/shapenet_renderer/ship_renders/*/*/rgb/*.png')
        # self.data = glob.glob('/scratch/s2576597/graf_datasets/ship_renders_train_upper_hemisphere_30_fov/*/rgb/*.png')
        self.data = glob.glob('/scratch/s2576597/pixel_nerf_datasets/ship_renders_train_upper_hemisphere_30_fov_v2_pixel_nerf/ships_train/*/rgb/*.png')


class ShapenetShipsTest(MyBaseDataset):
    def __init__(self, img_size, **kwargs):
        super().__init__(img_size)

        # self.data = glob.glob('/home/gitaar9/AI/TNO/shapenet_renderer/ship_renders/*/*/rgb/*.png')
        # self.data = glob.glob('/scratch/s2576597/graf_datasets/ship_renders_train_upper_hemisphere_30_fov/*/rgb/*.png')
        self.data = glob.glob('/scratch/s2576597/pixel_nerf_datasets/ship_renders_train_upper_hemisphere_30_fov_v2_pixel_nerf/ships_test/*/rgb/*.png')


class ShapenetCars(MyBaseDataset):
    def __init__(self, img_size, **kwargs):
        super().__init__(img_size)

        #self.data = glob.glob('/home/gitaar9/AI/TNO/shapenet_renderer/ship_renders/*/*/rgb/*.png')
        #self.data = glob.glob('/data/s2576597/graf_datasets/car_renders_train_upper_hemisphere_30_fov/*/rgb/*.png')
        self.data = glob.glob('/scratch/s2576597/pixel_nerf_datasets/car_renders_train_upper_hemisphere_30_fov_pixel_nerf/cars_train/*/rgb/*.png')


class ShapenetCarsTest(MyBaseDataset):
    def __init__(self, img_size, **kwargs):
        super().__init__(img_size)

        self.data = glob.glob('/scratch/s2576597/pixel_nerf_datasets/car_renders_train_upper_hemisphere_30_fov_pixel_nerf/cars_test/*/rgb/*.png')


class CARLA(MyBaseDataset):
    def __init__(self, img_size, **kwargs):
        super().__init__(img_size)

        # self.data = glob.glob('/home/gitaar9/AI/TNO/graf/data/carla/*.png')
        self.data = glob.glob('/data/s2576597/graf_datasets/carla/*.png')
        print("DATASET SIZE: ", len(self.data))


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3


def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):

    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=16,
    )

    return dataloader, 3
