from Model_score import SCORE_CNN
from Model_obj import OBJ_CNN     # obj CNN construction
import util
import dataset
import properties
import cnn
import Hypothesis
import TYPE
import Model_score
import torch
import numpy as np
import math


def getRandHyp(gaussRot, gaussTrans):
    trans = np.array([np.random.normal(0, gaussTrans), np.random.normal(0, gaussTrans), np.random.normal(0, gaussTrans)])
    rotAxis = np.array(([np.random.rand(), np.random.rand(), np.random.rand()]))
    rotAxis = rotAxis/np.linalg.norm(rotAxis)
    rotAxis = rotAxis * np.random.normal(0, gaussRot) * math.pi/180
    # Construct rot vector and translation vector
    RotVec = np.zeros(6)
    RotVec[:3] = rotAxis
    RotVec[3:6] = trans
    # Construct a hypothesis
    h = Hypothesis.Hypothesis()
    h.RodvecandTrans(RotVec)
    return h


def assembleData(imageCount, hypsPerImage, objInputSize, rgbInputSize, dataset, model, temperature):
    data = []
    label = []
    gp = properties.GlobalProperties()
    camMat = gp.getCamMat()
    for i in range(imageCount):
        imgIdx = np.random.randint(0, dataset.size())
        imgBGR = dataset.getBGR(imgIdx)
        info = dataset.getInfo(imgIdx)
        sampling = cnn.stochasticSubSample(imgBGR, objInputSize, rgbInputSize)
        # Through the trained network, get the estimated object coordinate
        estObj = cnn.getCoordImg(imgBGR, sampling, rgbInputSize, model)
        poseGT = Hypothesis.Hypothesis()
        poseGT.Info(info)
        for h in range(hypsPerImage):
            driftLevel = np.random.randint(0, 3)
            if not driftLevel:
                poseNoise = poseGT * getRandHyp(2, 2)
            else:
                poseNoise = poseGT * getRandHyp(10, 100)
            # Construct data and label
            # input: reprojection error image
            data.append(cnn.getDiffMap(TYPE.our2cv([poseNoise.getRotation(), poseNoise.getTranslation()]),
                                       estObj, sampling, camMat))
            label.append(-1 * temperature * max(poseGT.calcAngularDistance(poseNoise),
                                               np.linalg.norm(poseGT.getTranslation() - poseNoise.getTranslation())))
    data = np.array(data)
    label = np.array(label)
    return data, label


def assembleBatch(offset, size, permutation, data, label):
    batchData = data[permutation[offset]:permutation[offset + size], :, :, :]
    batchLabels = label[permutation[offset]:permutation[offset + size], :]
    return batchData, batchLabels


if __name__ == '__main__':

    # Parameter setting
    trainingImages = 100
    trainingHyps = 16
    trainingRounds = 80
    objTemperature = 10
    objBatchSize = 64
    lrInitPre = 0.0001
    CNN_OBJ_PATCHSIZE = 40
    CNN_RGB_PATCHSIZE = 42

    # Training path setting
    dataDir = './'
    trainingDir = dataDir + 'training/'
    trainingSets = util.getSubPaths(trainingDir)

    # Initialize training dataset
    trainingDataset = dataset.Dataset()
    trainingDataset.readFileNames(trainingSets[0])
    trainingDataset.SetObjID(1)

    # Load RGB CNN's parameters
    RGB_NET = OBJ_CNN()
    RGB_NET.load_state_dict(torch.load('./Model parameter/obj_model_init.pkl'))

    # Construct SCORE CNN network
    SCORE_NET = SCORE_CNN()

    # Training parameter
    trainCounter = 0
    optimizer = torch.optim.Adam(SCORE_NET.parameters(), lrInitPre)

    # For recording
    trainfile = open('./Model parameter/training_loss_score.txt', 'a')

    # Iteration for training
    while trainCounter <= trainingRounds:
        SCORE_NET.train()
        data, label = assembleData(trainingImages, trainingHyps,
                                   CNN_OBJ_PATCHSIZE, CNN_RGB_PATCHSIZE,
                                   trainingDataset, RGB_NET, objTemperature)
        # For shuffle batch
        data_size = np.shape(data)[0]
        trainPermutation = np.linspace(0, data_size - 1, data_size, dtype=int)
        np.random.shuffle(trainPermutation)
        for i in range(int(data_size / objBatchSize)):
            batchData, batchLable = assembleBatch(i * objBatchSize, objBatchSize, trainPermutation, data, label)
            trainLoss = Model_score.train(SCORE_NET, batchData, batchLable, optimizer, trainCounter)
            trainCounter += 1
            np.savetxt(trainfile, np.array([trainCounter, trainLoss]))
            print('Training Loss:', trainLoss)

