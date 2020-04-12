import Hypothesis
import numpy as np
import properties
import cv2
import util
import dataset
import torch
import TYPE


# trainPermutation = np.linspace(0, 4, 5, dtype=int)
# np.random.shuffle(trainPermutation)
# print(trainPermutation)
# # Training path setting
# dataDir = './'
# trainingDir = dataDir + 'training/'
# trainingSets = util.getSubPaths(trainingDir)
#
# # Initialize training dataset
# trainingDataset = dataset.Dataset()
# trainingDataset.readFileNames(trainingSets[0])
# trainingDataset.SetObjID(1)
#
# imgIdx = np.random.randint(0, trainingDataset.size())
# imgBGR = trainingDataset.getBGR(imgIdx)
#
# width = np.shape(imgBGR)[1]
# height = np.shape(imgBGR)[0]
# inputSize = 42
# x = np.random.randint(inputSize/2, width - inputSize/2)
# y = np.random.randint(inputSize/2, height - inputSize/2)
# print('x', x)
# print('y', y)
# data = imgBGR[int(y - inputSize/2): int(y + inputSize/2), int(x - inputSize/2): int(x + inputSize/2), :]
# cv2.imshow('data', data)
# cv2.waitKey(0)


# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(a[3:7])
# rot = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
# trans = np.array([2, 3, 1])
#
# transformation = np.zeros([4, 4])
# transformation[:3, :3] = rot
# transformation[:3, 3] = trans
# transformation[3, 3] = 1
#
# print(cv2.Rodrigues(rot))

# file = open('/home/open/eth/2020spring/3DV/Project/C++/DSAC/7scenes/7scenes_chess/sensorTrans.dat')
# lines = file.readlines()
# for line in lines:
# #     print(np.array(list(map(float, line))))
#
# file = np.fromfile('/home/open/eth/2020spring/3DV/Project/C++/DSAC/7scenes/7scenes_chess/sensorTrans.dat', dtype=float)
# new_file = np.delete(file, 0).reshape(4, 4)
# print(new_file)
# rodandtrans = np.zeros(6)
# rodandtrans[:3] = cv2.Rodrigues(rot)
# rodandtrans[3:] = trans
# #rot and trans
# h = Hypothesis.Hypothesis()
# h.RotandTrans(rot, trans)
#
# #transformationMatrix
# h = Hypothesis.Hypothesis()
# h.TransformationMatrix(transformation)
#
# #Rodandtrans
# h = Hypothesis.Hypothesis()
# h.RodvecandTrans(rodandtrans)

# #points pair vector
# h = Hypothesis.Hypothesis()
# h.Points(points)


# path1 = '/home/open/eth/2020spring/3DV/Project/C++/DSAC/7scenes/7scenes_chess/test/scene/rgb_noseg/frame-000000.color.png'
# path2 = '/home/open/eth/2020spring/3DV/Project/C++/DSAC/7scenes/7scenes_chess/test/scene/rgb_noseg/frame-000001.color.png'
# img1 = cv2.imread(path1)
# img2 = cv2.imread(path2)
# height = np.shape(img1)[0]
# width = np.shape(img2)[1]
# img = [img1, img2]
# img = np.array(img)
# c = np.
# img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)))
# print(img1[1, 1, :])
# print(img[0, :, 1, 1])

# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = torch.from_numpy(a).type(torch.FloatTensor)
# print(b.size()[0])
# c = np.array([[1, 2, 2], [1, 2, 2]])
# d = torch.from_numpy(c).type(torch.FloatTensor)
# e = torch.div(b, d)
# print(e)



# img1 = img[:, :, 0]
# print(np.shape(img1))
# print(np.shape(img))
# print(np.shape(img)[0])
# # cv2.imshow('s', img)
# # cv2.waitKey(0)
# print(np.shape(img))
# a = np.array([1, 1])
# print(np.shape(a))
#
# a = np.array([[2, 3, 1], [2, 3, 3], [3, 2, 1]], dtype=np.float)
# rvec = np.array([2, 3, 1])
# b =np.dot(a, rvec)
# print(np.shape(b))
#
# transfile = 'translation.txt'
# file_trans = open(transfile)
# lines = file_trans.readlines()
# for line in lines:
#     print(line)

# a = np.zeros([4, 4, 3])
# b = np.ones([4, 4, 3])
# c = []
# c.append(a)
# c.append(b)
# print(type(c))
# d = np.array(c).reshape((2, 4, 4, 3))
# print(np.shape(d))
# print(d[0, :, :, :])
# print(d[1, :, :, :])

# path1 = '/home/open/eth/2020spring/3DV/Project/C++/DSAC/7scenes/7scenes_chess/test/scene/rgb_noseg/frame-000000.color.png'
# img = cv2.imread(path1)
# img2 = img[20:100, 20:100, :]
# img3 = img[40:120, 40:120, :]
# a = [img2, img3]
# b = np.array(a)
# print(np.shape(b))
# print(np.shape(img2))
# print(np.shape(img))

# a = np.random.normal(0, 2)
# b = np.ones(3)
# c = a * b
# print(c)
# print(b)

# a = np.random.randint(0, 3, size=2)
# print(a)
import time
from numba import jit, vectorize, int64, float32
import dataset


# @jit
# def get_img(img, imgDepth):
#     for x in range(np.shape(imgDepth)[1]):
#         for y in range(np.shape(imgDepth)[0]):
#             img[y, x, :] = dataset.pxToEye(x, y, imgDepth[y][x])
#     return img
#
# def test_numba(i):
#     time_start = time.time()
#     data = dataset.Dataset()
#     basepath = '/home/open/eth/2020spring/3DV/Project/Python/training/scene'
#     data.readFileNames(basepath)
#     imgDepth = data.getDepth(i)
#     img = np.zeros([np.shape(imgDepth)[0], np.shape(imgDepth)[1], 3])
#     img = get_img(img, imgDepth)
#     time_end = time.time()
#     print('with numba', time_end - time_start)
# #
# #
# def without_numba(i):
#     time_start = time.time()
#     data = dataset.Dataset()
#     basepath = '/home/open/eth/2020spring/3DV/Project/Python/training/scene'
#     data.readFileNames(basepath)
#     imgDepth = data.getDepth(i)
#     img = np.zeros([np.shape(imgDepth)[0], np.shape(imgDepth)[1], 3])
#     for x in range(np.shape(imgDepth)[1]):
#         for y in range(np.shape(imgDepth)[0]):
#             img[y, x, :] = dataset.pxToEye(x, y, imgDepth[y][x])
#     time_end = time.time()
#     print('without numba', time_end - time_start)
#
#
# if __name__ == '__main__':
#     test_numba(2)
#     without_numba(2)

import read_data
from pathos.multiprocessing import ProcessingPool as newpool
import multiprocessing as mp


def get_xy(width, height):
    x = np.array(range(width * height)) % width
    y = np.array(range(width * height)) // width
    return x, y


if __name__ == '__main__':
    time_start = time.time()
    i = 2
    data = dataset.Dataset()
    basepath = '/home/open/eth/2020spring/3DV/Project/Python/training/scene'
    data.readFileNames(basepath)
    img = read_data.readData_depth(basepath)
    time0 = time.time()
    print(time0 - time_start)
    imgDepth = data.getDepth(i)
    width = np.size(imgDepth, 0)
    height = np.size(imgDepth, 1)
    time1 = time.time()
    print(time1 - time_start)
    imgDepth_fla = imgDepth.flatten()
    time2 = time.time()
    print(time2 - time1)
    x, y = get_xy(width, height)
    time_end = time.time()
    print(time_end - time2)
    pool = newpool(mp.cpu_count())
    img = list(pool.map(dataset.pxToEye, x, y, imgDepth_fla))
    time3 = time.time()
    print(time3 - time_start)








