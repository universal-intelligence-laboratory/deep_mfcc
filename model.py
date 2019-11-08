import torch
from torchaudio.transforms import MFCC
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MFCCNet(nn.Module):
    def __init__(self):
        super(MFCCNet, self).__init__()
        self.mfcc = MFCC(sample_rate=8000, n_mfcc=39)

        # ,padding_mode='circular'
        self.c1 = nn.Conv1d(40*4, 39*4, kernel_size=5)
        self.c2 = nn.Conv1d(39*4, 39*2, kernel_size=5)
        self.c3 = nn.Conv1d(39*2, 39*2, kernel_size=5)
        # ,padding_mode='circular'
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 39, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1)

        self.d1 = nn.Linear(70*80, 16*81)

    def get_mfcc(self, x):
        B, H = tuple(x.shape)
        x = x.view(B, H)
        mfcc = self.mfcc(x)
        return mfcc

    def forward(self, x):
        # print(x.shape)
        B = tuple(x.shape)[0]
        mfcc = self.get_mfcc(x)
        B, H = tuple(x.shape)
        x = x.view(B, 40*4, -1)
        # print(x.shape)
        x = F.instance_norm(self.c1(x))
        # print(x.shape)
        x = F.instance_norm(self.c2(x))
        # print(x.shape)
        x = F.instance_norm(self.c3(x))
        # print(x.shape)
        x = x.view(B, 1, 78, -1)
        # print(x.shape)
        x = F.instance_norm(self.conv1(x))
        # print(x.shape)
        x = F.instance_norm(self.conv2(x))
        # print(x.shape)
        x = x.view(B,  39, -1)

        x = self.d1(x)
        x = x.view(B, -1, 39, 81)

        x = self.conv3(x)
        x = x.view(B, 39, -1)
        # print(mfcc.shape)
        loss = F.mse_loss(x, mfcc)
        return loss, x, mfcc


class YDataset(Dataset):
    """普通numpy dataset"""

    def __init__(self, y=None, frame_len=2 * 8000):
        self.size = len(y)
        self.frame_len = frame_len
        self.y = torch.from_numpy(y)
        print(self.y.shape)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < (self.size - self.frame_len):
            data = self.y[idx:idx+self.frame_len]
        else:
            data = self.y[idx-self.frame_len:idx]
        return data
