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


def train(model, data, label, optimizer, num, device):
    # from numpy to tensor
    mean = 45
    data = torch.from_numpy(np.transpose(data, (0, 3, 1, 2))) - mean
    label = torch.from_numpy(label)
    data = data.to(device)
    label = label.to(device)

    # set training mode
    model.train()

    # calculate prediction and loss
    criterion_loss = nn.PairwiseDistance(p=1).to(device)
    pred = model(data)
    loss = criterion_loss(pred, label)
    loss = loss / label.size()[0]
    loss.backward()

    # optimize
    optimizer.step()

    # num counter
    if not num % storeIntervalPre:
        torch.save(model.state_dict(), './Model parameter/obj_model_init.pkl')
    if not num % lrIntervalPre:
        for param in optimizer.param_groups:
            param['lrInitPre'] *= 0.5
    return loss