from torch.utils.data.dataset import Dataset
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


class CustomDataset1(Dataset):
    """
    The form of datasets consists of two 3-channels images with one corresponding rectified information
    """
    def __init__(self, root_dir, img_path, label_path, transform = None):
        self.root_dir = root_dir
        self.img_path = root_dir + img_path
        self.img_list = sorted(os.listdir(self.img_path))
        self.label_path = root_dir + label_path
        self.size = int(len(os.listdir(self.img_path))/2)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        file = open(self.label_path)
        lines = file.readlines()
        KEYWORD = self.img_list[item]
        imgR = io.imread(self.img_path + KEYWORD)
        imgR = self.transform(imgR)
        imgL = io.imread(self.img_path + self.img_list[item + self.size])
        imgL = self.transform(imgL)
        for line in lines:
            if KEYWORD in line:
                line_rec = line.lstrip('\n').split()
                # info = np.array(list(map(float, [line_rec[2], line_rec[3]])))
                info = np.array(list(map(float, [line_rec[2], line_rec[3]]))) * 100
        return imgR, imgL, info


class CustomDataset2(Dataset):
    """
    The form of datasets consists of one 6-channel matrix with one corresponding rectified information
    """
    def __init__(self, root_dir, img_path, label_path, transform = None):
        self.root_dir = root_dir
        self.img_path = root_dir + img_path
        self.img_list = sorted(os.listdir(self.img_path))
        self.label_path = root_dir + label_path
        self.size = int(len(os.listdir(self.img_path))/2)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        file = open(self.label_path)
        lines = file.readlines()
        KEYWORD = self.img_list[item]
        img1 = io.imread(self.img_path + KEYWORD)
        img2 = io.imread(self.img_path + self.img_list[item + self.size])
        height = np.shape(img1[:, :, 0])[0]
        width = np.shape(img1[:, :, 0])[1]
        image = np.zeros([height, width, 6])
        for line in lines:
            if KEYWORD in line:
                line_rec = line.lstrip('\n').split()
                info = np.array(list(map(float, [line_rec[2], line_rec[3]])))
        return image, info








