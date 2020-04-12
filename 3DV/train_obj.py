import dataset
import util
import numpy as np
import cv2
import torch
from Model_obj import OBJ_CNN
import Model_obj
import Customized_Datasets
import time
from torchvision import transforms

if __name__ == '__main__':

    # Parameters setting
    inputSize = 42
    channels =3
    trainingLimit = 300000
    trainingImages = 100
    trainingPatches = 512
    BATCHSIZE = 64
    lrInitPre = 0.0001
    storeIntervalPre = 1000
    lrInterval = 50000
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training path setting
    dataDir = './'
    trainingDir = dataDir + 'training/'
    trainingSets = util.getSubPaths(trainingDir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Initialize training dataset
    training = dataset.Dataset()
    training.readFileNames(trainingSets[0])
    training.SetObjID(1)
    TrainData = Customized_Datasets.RGB_DATASET(training, inputSize, transform)

    # Check if training set is empty
    if not training.size():
        print('The training set is empty !')

    # Construction Model
    OBJ_NET = OBJ_CNN().to(DEVICE)
    OBJ_NET.train()

    # Training
    trainCounter = 0
    round = 0
    StoreCounter = 0
    optimizer = torch.optim.Adam(OBJ_NET.parameters(), lrInitPre)

    # For recording
    trainfile = open('./Model parameter/training_loss_obj.txt', 'a')

    # Iteration for training
    while trainCounter <= trainingLimit:
        train_loader = torch.utils.data.DataLoader(TrainData, batch_size=BATCHSIZE, shuffle=True, num_workers=8)
        loss = Model_obj.train(OBJ_NET, train_loader, optimizer, trainCounter, DEVICE)
        print(type(loss))
        trainCounter += 1
        np.savetxt(trainfile, np.array([trainCounter, loss]))
        print('Training Loss:', loss)







