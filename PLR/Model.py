import torch
import torch.nn as nn


#CNN Construction
class Miscalibration1(nn.Module):
    """
    This CNN is for two image with 3-channels
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.FC1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=2*128*4*11, out_features=2048),   # 2:two images merge   128:num of channels   4:height   11:width
            nn.ReLU(),)
        self.FC2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),)
        self.FC3 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2),
            nn.Linear(in_features=2, out_features=2),
        )

    def forward(self, x1, x2):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)
        x1 = self.conv7(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        x2 = self.conv6(x2)
        x2 = self.conv7(x2)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), 1)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        return x


class Miscalibration2(nn.Module):
    """
    This CNN is for one matrix with 6-channels
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, ),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )
        self.FC1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=128*4*10, out_features=2048),   # 128:num of channels   4:height   10:width
            nn.ReLU(),)
        self.FC2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),)
        self.FC3 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        return x

