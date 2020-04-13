import torch.nn as nn
import torch
import numpy as np


storeIntervalPre = 100
lrIntervalPre = 5000

class SCORE_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1,),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, ),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0, ),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, ),
            nn.ReLU(),
        )
        self.FC = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        return x


def train(model, train_loader, device, lossfunc, optimizer, batchsize, num):
    for idx, (BatchData, BatchLabel) in enumerate(train_loader):
        BatchData, BatchLabel, lossfunc = BatchData.to(device), BatchLabel.to(device), lossfunc.to(device)
        # Forward
        pred = model(BatchData)
        loss = lossfunc(pred, BatchLabel)/batchsize
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num += 1
        # num counter
        if not num % storeIntervalPre:
            torch.save(model.state_dict(), './Model parameter/score_model_init.pkl')
        if not num % lrIntervalPre:
            for param in optimizer.param_groups:
                param['lr'] *= 0.5
    return loss, num


def forward(model, data, transform):
    # From numpy to tensor
    data_tensor = transform(data)
    pred = model(data_tensor)
    return pred.numpy()