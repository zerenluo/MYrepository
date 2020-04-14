from Model_score import SCORE_CNN
from Model_obj import OBJ_CNN     # obj CNN construction
from Customized_Datasets import SCORE_DATASET
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
from torchvision import transforms
import time


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
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training path setting
    dataDir = './'
    trainingDir = dataDir + 'training/'
    trainingSets = util.getSubPaths(trainingDir)

    # Load RGB CNN's parameters
    RGB_NET = OBJ_CNN()
    RGB_NET.load_state_dict(torch.load('./Model parameter/obj_model_init.pkl'))

    # Dataset process
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Initialize training dataset
    trainingDataset = dataset.Dataset()
    trainingDataset.readFileNames(trainingSets[0])
    trainingDataset.SetObjID(1)
    TrainData = SCORE_DATASET(trainingDataset, objInputSize=CNN_OBJ_PATCHSIZE, rgbInputSize=CNN_RGB_PATCHSIZE,
                              model=RGB_NET, temperature=objTemperature, transform=transform)

    # Construct SCORE CNN network
    SCORE_NET = SCORE_CNN()

    # Training parameter
    trainCounter = 0
    round = 0
    optimizer = torch.optim.Adam(SCORE_NET.parameters(), lrInitPre)
    lossfunction = torch.nn.PairwiseDistance(p=1)

    # For recording
    loss_list = []
    time_start = time.time()

    # Iteration for training
    while round <= trainingRounds:
        # Print round info
        round += 1
        print('Starting Round:', round)
        # Load datasets and train
        SCORE_NET.train()
        train_loader = torch.utils.data.DataLoader(TrainData, batch_size=objBatchSize, shuffle=True)
        loss, trainCounter = Model_score.train(model=SCORE_NET, train_loader=train_loader, batchsize=objBatchSize,
                                               lossfunc=lossfunction, optimizer=optimizer, device=DEVICE, num=trainCounter)
        # Recording
        loss_list.append(loss)
        time_end = time.time()
        print(loss)
        print('Time Cost:', time_end - time_start)





