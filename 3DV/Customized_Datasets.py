from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import math
import cnn
import TYPE
from Hypothesis import Hypothesis
from properties import GlobalProperties


class RGB_DATASET(Dataset):
    def __init__(self, dataset, inputSize, transform):
        self.dataset = dataset
        self.inputSize = inputSize
        self.imgIdx = np.random.randint(0, dataset.size())
        self.imgBGR = dataset.getBGR(self.imgIdx)
        self.imgObj = dataset.getObj(self.imgIdx)
        self.transform = transform

    def __len__(self):
        return 51200

    def __getitem__(self, item):

        width = np.size(self.imgBGR, 0)
        height = np.size(self.imgBGR, 1)
        x = np.random.randint(self.inputSize / 2, width - self.inputSize / 2)
        y = np.random.randint(self.inputSize / 2, height - self.inputSize / 2)
        data = self.imgBGR[int(y - self.inputSize/2): int(y + self.inputSize/2),
               int(x - self.inputSize/2): int(x + self.inputSize/2), :]
        data_trans = np.transpose(data, (2, 0, 1))
        img = self.transform(data_trans)
        label = self.imgObj[y, x]/1000.0
        return img, label


# Function for SCORE DATASET
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
    h = Hypothesis()
    h.RodvecandTrans(RotVec)
    return h

class SCORE_DATASET(Dataset):
    def __init__(self, dataset, objInputSize, rgbInputSize, model, temperature, transform):
        self.dataset = dataset
        self.objInputSize = objInputSize
        self.rgbInputSize = rgbInputSize
        self.imgIdx = np.random.randint(0, dataset.size())
        self.imgBGR = dataset.getBGR(self.imgIdx)
        self.info = dataset.getInfo(self.imgIdx)
        self.model = model
        self.temperature = temperature


    def __len__(self):
        return 1600

    def __getitem__(self, item):
        # Get parameter
        gp = GlobalProperties()
        camMat = gp.getCamMat()
        # Sampling for reprojection error image
        sampling = cnn.stochasticSubSample(self.imgBGR, self.objInputSize, self.rgbInputSize)
        estObj = cnn.getCoordImg(self.imgBGR, sampling, self.rgbInputSize, self.model)
        # Produce GroundTruth Label
        poseGT = Hypothesis()
        poseGT.Info(self.info)
        driftLevel = np.random.randint(0, 3)
        if not driftLevel:
            poseNoise = poseGT * getRandHyp(2, 2)
        else:
            poseNoise = poseGT * getRandHyp(10, 100)
        data = cnn.getDiffMap(TYPE.our2cv([poseNoise.getRotation(), poseNoise.getTranslation()]),
                              estObj, sampling, camMat)
        label = -1 * self.temperature * max(poseGT.calcAngularDistance(poseNoise),
                                            np.linalg.norm(poseGT.getTranslation() - poseNoise.getTranslation())/10.0)
        return data, label
