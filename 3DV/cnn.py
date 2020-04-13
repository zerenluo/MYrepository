import util
import properties
import TYPE
import Hypothesis
import numpy as np
import random
import cv2
import Model_obj
from torchvision import transforms


CNN_OBJ_MAXINPUT = 100.0


def Transform_OBJ():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform


def Transform_SCORE():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform


def containsNaNs(m):
    return np.any(np.isnan(m))


def entropy(dist):
    temp = -1*dist*np.log2(dist)
    return sum(temp)


def upperbound(num, prob):
    for i in range(len(prob)):
        if prob[i][0] > num: break
    return i


def draw(probs):
    cumProb = []
    probsum = 0
    for i in range(len(probs)):
        if probs[i] < np.finfo(float).eps: continue
        probsum += probs[i]
        cumProb.append([probsum, i])
    gp = properties.GlobalProperties()
    if gp.randomDraw:
        rand = random.uniform(0, probsum)
        print(rand)
        return upperbound(rand, cumProb)
    else:
        return np.argmax(probs)


def safeSolvePnP(objPts, imgPts, camMat, disCoeffs, methodFlag):
    """
    we should transfer the form of objPts and imgPts from list to np.array before using it
    :param objPts:
    :param imgPts:
    :param camMat:
    :param disCoeffs:
    :param methodFlag:
    :return:
    """
    retval, _, _ = cv2.solvePnP(objPts, imgPts, camMat, disCoeffs, None, None, None, methodFlag)
    if not retval:
        rvec = np.zeros(3)
        tvec = np.zeros(3)
    else:
        _, rvec, tvec = cv2.solvePnP(objPts, imgPts, camMat, disCoeffs, None, None, None, methodFlag)
    return [rvec, tvec]


def dPNP(imgPts, objPts, eps = 0.1):
    if len(imgPts) == 4:
        pnpMethod = cv2.SOLVEPNP_P3P
    else:
        pnpMethod = cv2.SOLVEPNP_ITERATIVE
    gp = properties.GlobalProperties()
    camMat = gp.getCamMat()
    imgPts = np.array(imgPts, np.int64)
    jacobean = np.zeros([6, len(objPts)*3])
    for i in range(len(objPts)):
        for j in range(3):
            # Forward step
            if j == 0: objPts[i][0] += eps
            elif j == 1: objPts[i][1] += eps
            elif j == 2: objPts[i][2] += eps
            objPts = np.array(objPts, np.float64)
            _, rot_f, tvec_f = safeSolvePnP(objPts, imgPts, camMat, None, pnpMethod)
            Trans_f = TYPE.cv2our([rot_f, tvec_f])
            h_f = Hypothesis.Hypothesis()
            h_f.RotandTrans(Trans_f[0], Trans_f[1])
            fstep = h_f.getRodVecAndTrans()

            # Backward step
            if j == 0: objPts[i][0] -= 2*eps
            elif j == 1: objPts[i][1] -= 2*eps
            elif j == 2: objPts[i][2] -= 2*eps
            objPts = np.array(objPts, np.float64)
            _, rot_b, tvec_b = safeSolvePnP(objPts, imgPts, camMat, None, pnpMethod)
            Trans_b = TYPE.cv2our([rot_b, tvec_b])
            h_b = Hypothesis.Hypothesis()
            h_b.RotandTrans(Trans_b[0], Trans_b[1])
            bstep = h_b.getRodVecAndTrans()

            # Back to normal state
            if j == 0: objPts[i][0] += eps
            elif j == 1: objPts[i][1] += eps
            elif j == 2: objPts[i][2] += eps

            # Gradient calculation
            for k in range(len(fstep)):
                jacobean[k][3*i+j] = (fstep[k] - bstep[k])/(2*eps)
            if containsNaNs(jacobean[:, 3*i+j]):
                return np.zeros([6, 3*objPts])
    return jacobean


def getAvg(mat):
    return np.average(np.abs(mat))


def getMax(mat):
    return np.max(np.abs(mat))


def getMed(mat):
    return np.median(np.abs(mat))


def getCoordImg(colorData, sampling, patchsize, model):
    patches = []
    modeImg = np.zeros([np.shape(sampling)[0], np.shape(sampling)[1], 3])
    width = np.shape(colorData)[1]
    height = np.shape(colorData)[0]
    for x in range(np.shape(modeImg)[1]):
        for y in range(np.shape(modeImg)[0]):
            origX = sampling[y][x][0]
            origY = sampling[y][x][1]
            if origX < patchsize/2 or origY < patchsize/2 or origX > width-patchsize/2 or origY > height-patchsize/2:
                continue
            minX = int(origX - patchsize/2)
            minY = int(origY - patchsize/2)
            maxX = int(origX + patchsize/2)
            maxY = int(origY + patchsize/2)
            patch = colorData[minY:int(maxY+1), minX:int(maxX+1), :]
            patches.append(patch)

    # Do prediction
    patches = np.array(patches)
    transform = Transform_OBJ()
    prediction = Model_obj.forward(model, patches, transform)
    for i in range(np.shape(prediction)[0]):
        x = int(i % np.shape(modeImg)[1])
        y = int(i / np.shape(modeImg)[0])
        modeImg[y, x, :] = prediction[i, :]
    return modeImg


def stochasticSubSample(inputMap, targetsize, patchsize):
    width = np.shape(inputMap)[1]
    height = np.shape(inputMap)[0]
    sampling = np.zeros(targetsize, targetsize, 2)
    xStride = (width - patchsize)/targetsize
    yStride = (height - patchsize)/targetsize
    xrange = np.zeros(targetsize)
    yrange = np.zeros(targetsize)
    for i in range(targetsize + 1):
        xrange[i] = patchsize/2 + i * xStride
        yrange[i] = patchsize/2 + i * yStride
    for x in range(targetsize):
        for y in range(targetsize):
            # using np.random.random() to substitute drand() temporarily
            sampling[y, x, 0] = int(xrange[i] + (xrange[i + 1]-xrange[i])*np.random.random())
            sampling[y, x, 1] = int(yrange[i] + (yrange[i + 1]-yrange[i])*np.random.random())
    return sampling


def getDiffMap(hyp, objectCoordinates, sampling, camMat):
    diffMap = np.zeros(np.shape(sampling))
    points3D = []
    points2D = []
    source2D = []
    for x in range(np.shape(sampling)[1]):
        for y in range(np.shape(sampling)[0]):
            points3D.append(objectCoordinates[y, x, :])
            points2D.append(sampling[y, x, :])
            source2D.append(np.array([x, y]))
    points3D_np = np.array(points3D)
    projections, _ = cv2.projectPoints(points3D_np, hyp[0], hyp[1], camMat, None)
    for i in range(len(projections)):
        curPt = points2D[i] - projections[i, :, :].reshape(2)
        diffMap[source2D[i][1]][source2D[i][0]] = min(np.linalg.norm(curPt), CNN_OBJ_MAXINPUT)
    diffMap = np.reshape(diffMap, (np.shape(sampling)[0], np.shape(sampling)[1], 1))
    return diffMap


def project(pt, obj, rot, t, camMat):
    f = camMat[0][0]
    ppx = camMat[0][2]
    ppy = camMat[1][2]
    objMat = np.dot(rot, obj)+t
    # Since we calculate the groudtruth object coordinate by using OPENCV coordinate
    # as can be seen in dataset.py 'getobj'
    px = -1 * f * objMat[0]/objMat[2] + ppx
    py = f * objMat[1]/objMat[2] + ppy
    pxy = np.array([px, py])
    return min(np.linalg.norm(pxy-pt), CNN_OBJ_MAXINPUT)


def dProjectObj(pt, obj, rot, t, camMat):
    f = camMat[0][0]
    ppx = camMat[0][2]
    ppy = camMat[1][2]
    objMat = np.dot(rot, obj) + t

    # Prevent division by zero
    if np.abs(objMat)[2] < np.finfo(float).eps: return np.zeros(3)
    px = -1 * f * objMat[0]/objMat[2] + ppx
    py = f * objMat[1]/objMat[2] + ppy
    pxy = np.array([px, py])

    # Calculate error
    err = np.linalg.norm(pxy-pt)
    if err > CNN_OBJ_MAXINPUT: return np.zeros(3)
    err += np.finfo(float).eps

    # derivative in x direction
    pxdx = -1 * f * rot[0][0]/objMat[2] + f * objMat[0] * rot[2][0] / (objMat[2]**2)
    pydx = f * rot[1][0]/objMat[2] - f * objMat[1] * rot[2][0] / (objMat[2]**2)
    dx = (pxy - pt)[0] * pxdx / err + (pxy - pt)[1] *pydx / err

    # derivative in y direction
    pxdy = -1 * f * rot[0][1] / objMat[2] + f * objMat[0] * rot[2][1] / (objMat[2] ** 2)
    pydy = f * rot[1][1] / objMat[2] - f * objMat[1] * rot[2][1] / (objMat[2] ** 2)
    dy = (pxy - pt)[0] * pxdy / err + (pxy - pt)[1] * pydy / err

    # derivative in z direction
    pxdz = -1 * f * rot[0][2] / objMat[2] + f * objMat[0] * rot[2][2] / (objMat[2] ** 2)
    pydz = f * rot[1][2] / objMat[2] - f * objMat[1] * rot[2][2] / (objMat[2] ** 2)
    dz = (pxy - pt)[0] * pxdz / err + (pxy - pt)[1] * pydz / err

    return np.array([dx, dy, dz])
