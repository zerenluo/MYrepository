# Avoid using ROS Python
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#
import skimage.io as io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os


dirpath_img = ['../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/',
                '../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_01/data/',
                '../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/',
                '../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/']
dirpath_parm = '../Raw Data/2011_09_26 _cal/calib_cam_to_cam.txt'

keyword = [['R_00:', 'T_00:', 'K_00:'],
           ['R_01:', 'T_01:', 'K_01:'],
           ['R_02:', 'T_02:', 'K_02:'],
           ['R_03:', 'T_03:', 'K_03:']]


R = np.zeros([3, 3, 4])
K = np.zeros([3, 3, 4])
T = np.zeros([3, 4])


def read_img():
    cam2 = []
    cam3 = []
    file_cam2 = sorted(os.listdir(dirpath_img[2]))
    file_cam3 = sorted(os.listdir(dirpath_img[3]))
    for file in file_cam2:
        cam2.append(os.path.join(dirpath_img[2], file))
    for file in file_cam3:
        cam3.append(os.path.join(dirpath_img[3], file))
    return cam2, cam3




# img0 = cv2.imread('../Raw Data/2011_09_26/2011_09_26_drive_0001_extract/image_00/data/0000000000.png', 0)
# img1 = cv2.imread('../Raw Data/2011_09_26/2011_09_26_drive_0001_extract/image_01/data/0000000000.png', 0)
# stereo = cv2.StereoBM_create(numDisparities=128, blockSize=7)
# disparity = stereo.compute(img0, img1)
# plt.imshow(disparity, 'gray')
# plt.show()


def read_exparam():
    f = open(dirpath_parm, 'r')
    lines = f.readlines()
    for line in lines:
        for i in range(4):
            if keyword[i][0] in line:
                line = line.lstrip(keyword[i][0])
                R[:, :, i] = np.array(list(map(float, line.split()))).reshape((3, 3))
            elif keyword[i][1] in line:
                line = line.lstrip(keyword[i][1])
                T[:, i] = np.array(list(map(float, line.split())))
            elif keyword[i][2] in line:
                line = line.lstrip(keyword[i][2])
                K[:, :, i] = np.array(list(map(float, line.split()))).reshape((3, 3))
    return 0


def get_rec():
    read_exparam()
    T_diff = T[:, 2]-T[:, 3]
    e1 = T_diff/np.linalg.norm(T_diff)
    e2 = np.array([-T_diff[1], T_diff[0], 0])
    e2 = e2/np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    R_rec = np.vstack((np.vstack((e1, e2)), e3))
    return R_rec


def get_homography(R_rec, R1, R2, K1, K2):
    h1 = np.dot(np.dot(K1, R_rec), np.linalg.inv(K1))
    R_rot = np.dot(np.linalg.inv(R1), R2)
    R_rec2 = np.dot(R_rot, R_rec)
    h2 = np.dot(np.dot(K2, R_rec2), np.linalg.inv(K2))
    return [h1, h2]


def img_rectify():
    R_rec = get_rec()
    [h1, h2] = get_homography(R_rec, R[:, :, 2], R[:, :, 3], K[:, :, 2], K[:, :, 3])
    [img0, img1] = read_img()
    im1 = cv2.warpPerspective(img0[33], h1, (1500, 600))
    im2 = cv2.warpPerspective(img1[33], h2, (1500, 600))
    return [im1, im2]







