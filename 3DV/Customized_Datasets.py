from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


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