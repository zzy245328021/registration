#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/4/1 20:57
# @Author  : Eric Ching
# encoding:
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import rotate
import torch
import random


def intensity_shift3d(volumes, factor):
    """channel first"""
    volume = volumes + factor
    volume = np.clip(volume, 0, 1.0)

    return volume


def random_padding_crop(volume, label, output_size, mode='constant'):
    len_x, len_y, len_z = volume.shape
    len_x_o, len_y_o, len_z_o = output_size
    if len_x < len_x_o:
        # need pad
        pad_len = len_x_o - len_x
        pad_before = np.random.randint(0, pad_len + 1)
        pad_after = pad_len - pad_before
        volume = np.pad(volume, [[pad_before, pad_after], [0, 0], [0, 0]], mode=mode)
        label = np.pad(label, [[pad_before, pad_after], [0, 0], [0, 0]], mode=mode)
    elif len_x > len_x_o:
        clip_x = len_x - len_x_o
        clip_x = np.random.randint(0, clip_x)
        volume = volume[clip_x:clip_x + len_x_o, :, :]
        label = label[clip_x:clip_x + len_x_o, :, :]
    if len_y < len_y_o:
        # need pad
        pad_len = len_y_o - len_y
        pad_before = np.random.randint(0, pad_len + 1)
        pad_after = pad_len - pad_before
        volume = np.pad(volume, [[0, 0], [pad_before, pad_after], [0, 0]], mode=mode)
        label = np.pad(label, [[0, 0], [pad_before, pad_after], [0, 0]], mode=mode)
    elif len_y > len_y_o:
        clip_y = len_y - len_y_o
        clip_y = np.random.randint(0, clip_y)
        volume = volume[:, clip_y:clip_y + len_y_o, :]
        label = label[:, clip_y:clip_y + len_y_o, :]
    if len_z < len_z_o:
        # need pad
        pad_len = len_z_o - len_z
        pad_before = np.random.randint(0, pad_len + 1)
        pad_after = pad_len - pad_before
        volume = np.pad(volume, [[0, 0], [0, 0], [pad_before, pad_after]], mode=mode)
        label = np.pad(label, [[0, 0], [0, 0], [pad_before, pad_after]], mode=mode)
    elif len_z > len_z_o:
        clip_z = len_z - len_z_o
        clip_z = np.random.randint(0, clip_z)
        volume = volume[:, :, clip_z:clip_z + len_z_o]
        label = label[:, :, clip_z:clip_z + len_z_o]

    return volume, label


def random_rescale3d(volume, label, scale_range=0.1):
    """3d 尺度变换"""
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range, 3)
    x_scale, y_scale, z_scale = scale_factor
    org_x_len, org_y_len, org_z_len = volume.shape
    x_len = np.round(org_x_len * x_scale)
    y_len = np.round(org_y_len * y_scale)
    z_len = np.round(org_z_len * z_scale)
    output_shape = (x_len, y_len, z_len)
    trans_volume = resize(volume, output_shape, order=2, preserve_range=True)
    trans_label = resize(label, output_shape, order=0, preserve_range=True)

    return trans_volume, trans_label

def random_rotate3d(volume, label, rotate_range=15):
    angles = np.random.uniform(0, rotate_range, 3)
    for axes in [(0, 1), (1, 2), (0, 2)]:
        if np.random.random() < 0.5 and angles[0] != 0:
            volume = rotate(volume, angle=angles[0], axes=axes, order=2, reshape=False)
            label = rotate(label, angle=angles[0], axes=axes, order=0, reshape=False)

    return volume, label

def ramdom_flip(volume1, label1, volume2, label2):
    for ax in range(3):
        if random.random() < 0.5:
            volume1 = np.flip(volume1, axis=ax)
            label1 = np.flip(label1, axis=ax)
            volume2 = np.flip(volume2, axis=ax)
            label2 = np.flip(label2, axis=ax)

    return volume1, label1, volume2, label2


class LPBA(Dataset):
    def __init__(self, path_list,
                 target_shape=(192, 192, 192),
                 return_label=False,
                 augment=True):
        self.path_list = path_list
        n = len(self.path_list)
        self.n_pairs = n * (n - 1)
        self.target_shape = target_shape
        self.return_label = return_label
        self.augment = augment

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        fix_idx = idx // (len(self.path_list) - 1)
        fix_path = self.path_list[fix_idx]
        move_idx = idx % (len(self.path_list) - 1)
        if move_idx >= fix_idx:
            move_idx = move_idx + 1
        move_path = self.path_list[move_idx]
        move_volume, move_label = self.read_sample(move_path[0], move_path[1])
        fix_volume, fix_label = self.read_sample(fix_path[0], fix_path[1])

        if self.augment:
            move_volume, move_label = self.aug_sample(move_volume, move_label)
            fix_volume, fix_label = self.aug_sample(fix_volume, fix_label)
            move_volume, move_label, fix_volume, fix_label = ramdom_flip(move_volume, move_label, fix_volume, fix_label)

        move_volume = np.expand_dims(move_volume, axis=0)
        move_label = np.expand_dims(move_label, axis=0)
        fix_volume = np.expand_dims(fix_volume, axis=0)
        fix_label = np.expand_dims(fix_label, axis=0)

        if not self.return_label:
            return (torch.tensor(fix_volume.copy(), dtype=torch.float),
                    torch.tensor(fix_label.copy(), dtype=torch.float),
                    torch.tensor(move_volume.copy(), dtype=torch.float),
                    torch.tensor(move_label.copy(), dtype=torch.float))
        else:
            return (torch.tensor(fix_volume.copy(), dtype=torch.float),
                    torch.tensor(fix_label.copy(), dtype=torch.float),
                    torch.tensor(move_volume.copy(), dtype=torch.float),
                    torch.tensor(move_label.copy(), dtype=torch.float))

    def read_sample(self, volume_path, label_path):
        volume = nib.load(volume_path).get_data()
        label = nib.load(label_path).get_data()
        volume = self.normlize(volume)

        return volume, label

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def aug_sample(self, volume, label):
        """
        Args:
            volumes: list of volumes
            mask: segmentation volume
        Ret:
            x, y: [channel, h, w, d]
        """

        volume, label = random_rescale3d(volume, label)
        volume, label = random_rotate3d(volume, label)
        # volume = intensity_shift3d(volume, np.random.uniform(-0.1, 0.1))
        volume, label = random_padding_crop(volume, label, self.target_shape)

        return volume, label


def split_ds(data_root, nfold=5, seed=42, select=0):
    volume_dir = os.path.join(data_root, 'LPBA40_nii')
    label_dir = os.path.join(data_root, 'LPBA40_label')
    volume_path_list = sorted(glob.glob(os.path.join(volume_dir,"*.nii")))
    label_path_list = sorted(glob.glob(os.path.join(label_dir,"*.nii")))
    print("Total ", len(volume_path_list), " volumes")
    path_idx = np.arange(len(volume_path_list))
    np.random.seed(seed)
    np.random.shuffle(path_idx)
    nfold_list = np.split(path_idx, nfold)
    val_path_list = []
    train_path_list = []
    for i, fold in enumerate(nfold_list):
        if i == select:
            for idx in fold:
                val_path_list.append((volume_path_list[idx], label_path_list[idx]))
        else:
            for idx in fold:
                train_path_list.append((volume_path_list[idx], label_path_list[idx]))

    print("length of train list is ", len(train_path_list))
    print("length of validation list is ", len(val_path_list))
    for path in train_path_list:
        print(path)
    return train_path_list, val_path_list


def get_loaders(cfg):
    train_path_list, val_path_list = split_ds(cfg.data_dir, nfold=cfg.n_split_folds, select=cfg.select, seed=cfg.seed)
    train_ds = LPBA(train_path_list, augment=False)
    val_ds = LPBA(val_path_list, augment=False, return_label=True)
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.n_workers, pin_memory=True,
                                  shuffle=True)
    loaders['val'] = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.n_workers, pin_memory=True,
                                shuffle=False)

    return loaders


if __name__ == "__main__":
    from config import LPBAConfig
    cfg = LPBAConfig()
    loaders = get_loaders(cfg)
