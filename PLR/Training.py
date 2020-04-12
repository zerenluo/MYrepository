from Model import Miscalibration1
from CustomDatasets import CustomDataset1
from torchvision import transforms, datasets
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn


def train(model, device, train_loader, optimizer, lossfunc, epoch):
    for batch_ind, (imgR, imgL, info) in enumerate(train_loader):
        imgR, imgL, info = imgR.to(device), imgL.to(device), info.to(device)
        imgR.requires_grad = True
        imgL.requires_grad = True
        info.requires_grad = True
        with torch.no_grad():
            pred = model(imgR, imgL)
        loss = lossfunc(pred, info)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not (batch_ind + 1) % 10:
            print('Train epoch: {} Batch_ind: {} LOSS: {}'.format(epoch, batch_ind + 1, loss.item()))
            print('one of predcitions: ', pred[2, :].cpu().numpy())
            print('one of info', info[2, :].cpu().detach().numpy())


def test(model, device, test_loader, lossfunc, epoch):
    model.eval()
    for batch_ind, (imgR, imgL, info) in enumerate(test_loader):
        imgR, imgL, info = imgR.to(device), imgL.to(device), info.to(device)
        with torch.no_grad():
            pred = model(imgR, imgL)
        loss = lossfunc(pred, info)
        if not (batch_ind + 1) % 10:
            print('Test epoch: {} Batch_ind: {} LOSS: {}'.format(epoch, batch_ind + 1, loss.item()))


def validation(model, device, vali_loader, file1, file2):
    model.eval()
    for batch_ind, (imgR, imgL, info) in enumerate(vali_loader):
        imgR, imgL, info = imgR.to(device), imgL.to(device), info
        with torch.no_grad():
            pred = model(imgR, imgL)

        np.savetxt(file1, pred.cpu().numpy())
        np.savetxt(file2, info.numpy())


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.xavier_uniform_(model.weight.data)
        nn.init.xavier_uniform_(model.weight.data)
    elif isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight, gain=1)


if __name__ == '__main__':

    # Hyper-Parameter
    Batchsize = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LearningRate = 0.0001
    EPOCH = 50
    requires_grad = True
    NET_PATH = 'Miscalibration.pkl'

    # Parameter in loading dataset
    train_root_dir = '../Training/'
    test_root_dir = '../Test/'
    vali_root_dir = '../Validation/'
    img_path = 'Image/'
    label_path = 'rectified information.txt'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Loading training datasets
    dataset_train = CustomDataset1(train_root_dir, img_path, label_path, transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=Batchsize, shuffle=True)

    # Loading test datasets
    dataset_test = CustomDataset1(test_root_dir, img_path, label_path, transform)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=Batchsize, shuffle=True)

    # Loading validation datasets
    dataset_vali = CustomDataset1(vali_root_dir, img_path, label_path, transform)
    vali_loader = torch.utils.data.DataLoader(dataset_test, batch_size=Batchsize, shuffle=False)

    # Loading CNN
    MISCALI_CNN = Miscalibration1().to(DEVICE)
    MISCALI_CNN.apply(weights_init)

    # Optimizer and loss
    optimizer = optim.Adam(MISCALI_CNN.parameters(), LearningRate)
    loss_func = torch.nn.MSELoss().to(DEVICE)

    # Iterations for training
    for i in range(EPOCH):
        train(MISCALI_CNN, DEVICE, train_loader, optimizer, loss_func, i)
        test(MISCALI_CNN, DEVICE, test_loader, loss_func, i)
        if i == EPOCH - 1:
            torch.save(MISCALI_CNN.state_dict(), NET_PATH)

    # For validation
    # VALI_CNN = Miscalibration1().to(DEVICE)
    # VALI_CNN.load_state_dict(torch.load('Miscalibration.pkl'))
    # file1 = open('../validation.txt', 'a')
    # file2 = open('../vali_info.txt', 'a')
    # validation(VALI_CNN, DEVICE, vali_loader, file1, file2)
    # file1.close()
    # file2.close()
