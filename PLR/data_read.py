import skimage.io as io
import cv2
import numpy as np
import math
import os

# Path of parameter file
dirpath_parm = '../Raw Data/2011_09_26 _cal/calib_cam_to_cam.txt'
# Path of Raw Image
dirpath_img = ['../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/',
                '../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_01/data/',
                '../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data',
                '../Synced Data/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/']
# Keyword for getting extrinsic parameters
keyword = [['R_00:', 'T_00:', 'K_00:', 'D_00:', 'R_rect_00:'],
           ['R_01:', 'T_01:', 'K_01:', 'D_01:', 'R_rect_01:'],
           ['R_02:', 'T_02:', 'K_02:', 'D_02:', 'R_rect_02:'],
           ['R_03:', 'T_03:', 'K_03:', 'D_03:', 'R_rect_03:']]
# Initialization for parameters
R = np.zeros([3, 3, 4])
K = np.zeros([3, 3, 4])
R_rec = np.zeros([3, 3, 4])
T = np.zeros([3, 4])
D = np.zeros([5, 4])


def read_exparam():
    """
    Read raw parameters for both intrinsic and extrinsic parameters
    :return:
    """
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
            elif keyword[i][3] in line:
                line = line.lstrip(keyword[i][3])
                D[:, i] = np.array(list(map(float, line.split())))
            elif keyword[i][4] in line:
                line = line.lstrip(keyword[i][4])
                R_rec[:, :, i] = np.array(list(map(float, line.split()))).reshape((3, 3))
    return [R, K, T, D, R_rec]


def read_img():
    """
    Read Raw Image
    :return:
    """
    cam2 = []
    cam3 = []
    file_cam2 = sorted(os.listdir(dirpath_img[2]))
    file_cam3 = sorted(os.listdir(dirpath_img[3]))
    for file in file_cam2:
        cam2.append(os.path.join(dirpath_img[2], file))
    for file in file_cam3:
        cam3.append(os.path.join(dirpath_img[3], file))
    return cam2, cam3


def set_threshold(R_set, T_set):
    """
    Set randomly sampling threshold according to raw rotation angles and translation between cam2 and cam3
    :param R_set: rotation set
    :param T_set: translation set
    :return:
    """
    R_diff = np.dot(np.linalg.inv(R_set[:, :, 2]), R_set[:, :, 3])
    Rot_vec = cv2.Rodrigues(R_diff)[0]
    Rot_threshold = 0.1*np.linalg.norm(Rot_vec)
    Trans_diff = T_set[:, 2]-T_set[:, 3]
    Trans_threshold = 0.1*np.linalg.norm(Trans_diff)
    return [Rot_threshold, Trans_threshold]


def rand_transformation(rot, trans, R_threshold, T_threshold, ROTATION = True, TRANSLATION = True):
    """
    Fixed cam2, and randomly change rotation and translation of cam3
    :param rot: raw rotation matrix needed to be changed
    :param trans: raw translation vector needed to be changed
    :param R_threshold:
    :param T_threshold:
    :param ROTATION: whether change rotation
    :param TRANSLATION: whether change translation
    :return:
    """
    if ROTATION:
        rot_vec = cv2.Rodrigues(rot)[0]
        new_rot_vec = rot_vec + R_threshold*np.random.rand(3, 1)
        new_rot = cv2.Rodrigues(new_rot_vec)[0]
    if TRANSLATION:
        new_trans_vec = trans + T_threshold*np.random.rand(3)
    return [new_rot, new_trans_vec]





