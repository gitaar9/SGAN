import glob
import os
import shutil
import torch
import math

from pytorch_fid import fid_score
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
import datasets
from tqdm import tqdm
import copy
import argparse
import curriculums
import shutil


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str)
    opt = parser.parse_args()

    fake_dir = os.path.join(opt.eval_dir, 'fake_samples')
    real_dir = os.path.join(opt.eval_dir, 'real_samples')

    # fake_dir = real_dir
    real_dir = '/samsung_hdd/Files/AI/TNO/remote_folders/fid_evaluation/sgan_real_carla'

    metrics_dict = calculate_metrics(fake_dir, real_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
    # print(metrics_dict)
    fid = metrics_dict['frechet_inception_distance']
    kid = metrics_dict['kernel_inception_distance_mean'] * 100
    kid_std = metrics_dict['kernel_inception_distance_std'] * 100
    inception_score = metrics_dict['inception_score_mean']
    print('\n', opt.eval_dir.split('/')[-2])
    print(f"{fid:.2f} & {kid:.2f} $\\pm$ {kid_std:.2f} & ? & {inception_score:.2f}")

    # fid2 = fid_score.calculate_fid_given_paths([real_dir, fake_dir], 64, 'cuda', 2048)
    # print(f'Other fid: {fid2}')
