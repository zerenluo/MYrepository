import numpy as np
import os
from skimage import io
import torch
import cv2
import data_read


# file = open('../test.txt', 'a')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.ones([10, 2]).to(DEVICE)
# a = np.random.randint(0, 10)
# print(x[a, :].cpu().numpy())
# np.savetxt(file, x.numpy())
# file.close()
# file = open('../Training/rectified information.txt')
# lines = file.readlines()
# for line in lines:
#     line_rec = line.strip('\n').split()
#     info = np.array(list(map(float, [line_rec[2], line_rec[3]])))
#

# x = x.view(x.size(0), -1)
# y = torch.ones(2, 3, 3)
# y = y.view(y.size(0), -1)
# z = torch.cat((x, y), 1)
# print(z)

# print(torch.cuda.is_available())

# -*- coding: utf-8 -*-

import torch
import torch.optim as optim



# path = '../Rectified Image/Image/image02_000000000.png'
# img = io.imread(path)
# print(type(img))
# print(np.shape(img))
# file1 = sorted(os.listdir(path))[0]
# file2 = sorted(os.listdir(path))[1]
# img1 = io.imread(path+file1)
# img2 = io.imread(path+file2)
# print(np.shape(img1[:, :, 0]))
# file = open(path)
# lines = file.readlines()
# for line in lines:
#     line_rec = line.lstrip('\n').split()
#     print(np.array(list(map(float, [line_rec[2], line_rec[3]]))))
# a = 'ssss'
# b = 0.04324233424332
# f = float('%.5e' %b)
# print(f)
# c = np.array([a, f], dtype=None)
# d = c.reshape([1, 2])
# print(d)
# print(d[0][1])
# print(d[0][1].dtype)
# # np.savetxt(file, c.reshape([1, 2]), fmt='%s')

#
# path = '/home/open/eth/2020spring/PLR/Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data0000000000.png'
# img1 = cv2.imread(path)
# img2 = io.imread(path)
# cv2.imshow('s1', img1)
# cv2.imshow('s2', img2)
# cv2.waitKey(0)

[R, K, T, D, R_rec] = data_read.read_exparam()


print(np.dot(R[:, :, 0], R_rec[:, :, 0]))
print(np.dot(R[:, :, 1], R_rec[:, :, 1]))
print(np.dot(R[:, :, 2], R_rec[:, :, 2]))
print(np.dot(R[:, :, 3], R_rec[:, :, 3]))

