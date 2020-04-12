import torch.nn as nn
import torch
import numpy as np
import math


# Parameter Setting
storeIntervalPre = 1000
lrInterval = 50000
batchSize = 1600


class OBJ_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0,),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, ),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, ),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0, ),
            nn.ReLU(),
        )
        self.FC = nn.Sequential(
            nn.Linear(in_features=2*2*512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=3),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        return x


def criterion_loss(pred, label):
    return torch.mean(torch.norm(pred - label, 2, 1))


def train(model, train_loader, optimizer, num, device):
    for idx, (BatchData, BatchLabel) in enumerate(train_loader):
        BatchData, BatchLabel = BatchData.to(device), BatchLabel.to(device)
        # calculate prediction and loss
        pred = model(BatchData)
        loss = criterion_loss(pred, BatchLabel)
        loss.backward()
        # optimize
        optimizer.step()
        # num counter
    if not num % storeIntervalPre:
        torch.save(model.state_dict(), './Model parameter/obj_model_init.pkl')
    if not num % lrInterval:
        for param in optimizer.param_groups:
            param['lrInitPre'] *= 0.5
    return loss


def forward(model, data):
    # From numpy to tensor
    mean = 127
    data = torch.from_numpy(np.transpose(data, (0, 3, 1, 2))) - mean
    pred = model(data)
    return pred.numpy()




    



