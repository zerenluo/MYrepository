import Hypothesis
import util
import properties
import read_data
import TYPE
import numpy as np
import cv2


def pxToEye(x, y, depth):
    eye = np.zeros(3)
    if depth == 0:
        return eye
    gp = properties.GlobalProperties()
    eye[0] = (x-gp.imageWidth/2.0-gp.xShift)/(gp.focalLength/depth)
    eye[1] = -1*(y-gp.imageHeight/2.0-gp.yShift)/(gp.focalLength/depth)
    eye[2] = -1*depth
    return eye


def onObj(pt):
    return(any(pt))


class Dataset(object):
    def __init__(self, bgrFiles=[], depthFiles=[], infoFiles=[], objID=[]):
        self.bgrFiles = bgrFiles
        self.depthFiles = depthFiles
        self.infoFiles = infoFiles
        self.objID = objID

    def readFileNames(self, basePath):
        bgrPath = basePath + '/rgb_noseg/'
        bgrSuf = '.png'
        dPath = basePath + '/depth_noseg/'
        dSuf = '.png'
        infoPath = basePath + '/poses/'
        infoSuf = '.txt'
        self.bgrFiles = util.getFiles(bgrPath, bgrSuf)
        self.depthFiles = util.getFiles(dPath, dSuf)
        self.infoFiles = util.getFiles(infoPath, infoSuf, True)

    def SetObjID(self, objid):
        self.objID = objid

    def mapDepthTORGB(self, x, y, depth):
        gp = properties.GlobalProperties()
        eye = np.ones(4)
        eye[0] = (x-gp.imageWidth/2.0-gp.rawXShift)/(gp.secondaryFocalLength/float(depth))
        eye[1] = -1*(y-gp.imageHeight/2.0-gp.rawYShift)/(gp.secondaryFocalLength/float(depth))
        eye[2] = -1*depth
        eye = np.dot(gp.sensorTrans, eye)
        pix = np.zeros(2)
        pix[0] = int(eye[0]*gp.focalLength/float(depth)+gp.imageWidth/2.0+gp.xShift+0.5)
        pix[1] = int(-1*eye[1]*gp.focalLength/float(depth)+gp.imageHeight/2.0+gp.yShift+0.5)
        pix_int = pix.astype(int)
        return pix_int

    def getObjID(self):
        return self.objID

    def size(self):
        return len(self.bgrFiles)

    def getFileName(self, i):
        return self.bgrFiles[i]

    def getInfo(self, i):
        return read_data.readData_info(self.infoFiles[i])

    def getBGR(self, i):
        return read_data.readData_bgr(self.bgrFiles[i])

    def getDepth(self, i):
        img = read_data.readData_depth(self.depthFiles[i])
        gp = properties.GlobalProperties()
        if gp.rawData:
            depthMapped = np.zeros(np.shape(img))
            for x in range(np.shape(img)[1]):
                for y in range(np.shape(img)[0]):
                    depth = img[y][x]
                    if depth == 0: continue
                    pix = self.mapDepthTORGB(x, y, depth)
                    depthMapped[pix[1]][pix[0]] = depth
            img = depthMapped.astype('uint16')
        return img

    def getBGRD(self, i):
        img = TYPE.imag_brgd_t()
        img.bgr = self.getBGR(i)
        img.depth = self.getDepth(i)

    def getObj(self, i):
        depthData = self.getDepth(i)
        poseData = self.getInfo(i)
        h = Hypothesis.Hypothesis()
        h.Info(poseData)
        img_cam = np.zeros([np.shape(depthData)[0], np.shape(depthData)[1], 3])
        img_obj = np.zeros([np.shape(depthData)[0], np.shape(depthData)[1], 3])
        for x in range(np.shape(depthData)[1]):
            for y in range(np.shape(depthData)[0]):
                if not depthData[y][x]:
                    img_cam[y][x][:] = np.zeros(3)
                    continue
                img_cam[y][x][:] = pxToEye(x, y, depthData[y][x])
                img_obj[y][x][:] = h.invTransform(img_cam[y][x][:])
        return img_obj

    def getEye(self, i):
        imgDepth = self.getDepth(i)
        img = np.zeros([np.shape(imgDepth)[0], np.shape(imgDepth)[1], 3])
        for x in range(np.shape(imgDepth)[1]):
            for y in range(np.shape(imgDepth)[0]):
                img[y][x][:] = pxToEye(x, y, imgDepth[y][x])
        return img


# path = '/home/open/eth/2020spring/3DV/Project/C++/DSAC/7scenes/7scenes_chess/training/'
# dpath = util.getSubPaths(path)[0]
# print(dpath)
# d = Dataset()
# d.readFileNames(dpath)
# print(d.bgrFiles[0])

# d = Dataset()
# path = '/home/open/eth/2020spring/3DV/Project/C++/DSAC/7scenes/7scenes_chess/test/scene'
# d.readFileNames(path)
# pathd = path+'/depth_noseg/frame-000002.depth.png'
# img = read_data.readData_depth(pathd)
# print(img[20][200])
# b = d.getDepth(2)
#
# print('b', b[39][205])
# print(b.dtype)
# print(b.astype('uint16').dtype)
# cv2.imwrite('/home/yzy/Pictures/xx.png', b.astype('uint16'))
# cv2.imshow('s', b)
# cv2.waitKey(0)

# print(d.getDepth(2)[480][20])
# print(len(d.bgrFiles))

# print('map', d.mapDepthTORGB(200, 20, 2274))
# print('raw', img[205][39])

#
# c = cv2.imread('/home/yzy/Pictures/xx.png', -1)
# print(c[39][205])