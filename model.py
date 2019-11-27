import torch
from torchaudio.transforms import MelSpectrogram
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MFCCNet(nn.Module):
    def __init__(self):
        super(MFCCNet, self).__init__()
        self.mfcc = MelSpectrogram(sample_rate=8000)

        # ,padding_mode='circular'
        self.c1 = nn.Conv1d(20*4, 64, kernel_size=37)
        self.c2 = nn.Conv1d(64, 64, kernel_size=37)
        self.c3 = nn.Conv1d(64, 64, kernel_size=37)
        # ,padding_mode='circular'
        self.conv1 = nn.Conv2d(1, 10, kernel_size=37)
        self.conv2 = nn.Conv2d(10, 32, kernel_size=37)
        self.conv3 = nn.Conv2d(3, 1, kernel_size=1)

        self.d1 = nn.Linear(8192, 3*128*81)

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
        x = x.view(B, 20*4, -1)
        x = self.c1(x)
        x = self.c2(x)
        x = x.view(B, -1)
        x = self.d1(x)
        x = x.view(B, -1, 128, 81)

        x = self.conv3(x)
        x = x.view(B, 128, -1)
        loss = F.mse_loss(x, mfcc)
        return loss, x, mfcc


class MelNet(nn.Module):
    def __init__(self):
        super(MelNet, self).__init__()
        self.mfcc = MelSpectrogram(sample_rate=8000)

        self.c1 = nn.Conv1d(1, 128, kernel_size=400,stride=200,padding=200)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=37,padding=18)

    def get_mfcc(self, x):
        B, H = tuple(x.shape)
        x = x.view(B, H)
        mfcc = self.mfcc(x)
        return mfcc

    def forward(self, x):
        B = tuple(x.shape)[0]
        mfcc = self.get_mfcc(x)
        # print(mfcc.shape)
        B, H = tuple(x.shape)
        x = x.view(B, 1, -1)
        # print(x.shape)
        x = self.c1(x)
        # print(x.shape)
        x = x.view(B,128, -1)
        
        loss = F.smooth_l1_loss(x, mfcc)
        return loss, x, mfcc

class SampleBlock(nn.Module):
    def __init__(self, in_unit=128, out_unit=128, stride=1,padding=1):
        super(SampleBlock, self).__init__()
        self.stride = stride
        self.c1 = nn.Conv1d(in_unit, out_unit, kernel_size=3,stride = stride,padding=padding)
        self.bn = nn.BatchNorm1d(out_unit)
        self.act = nn.ReLU()
        if self.stride == 1:
            self.max = nn.MaxPool1d(3,stride=3)
        
    def forward(self, x):
        # print(x.shape)
        x = self.c1(x)
        x = self.bn(x)
        x = self.act(x)
        if self.stride == 1:
            x = self.max(x)
        # print(x.shape)
        return x
        

class SampleMel(nn.Module):
    def __init__(self):
        super(SampleMel, self).__init__()
        self.mfcc = MelSpectrogram(sample_rate=8000)

        self.sb1 = SampleBlock(in_unit=1,stride=3)
        self.sb2 = SampleBlock()
        self.sb3 = SampleBlock()
        self.sb4 = SampleBlock()
        self.sb5 = SampleBlock()
        self.sb6 = SampleBlock()
        self.sb7 = SampleBlock()


    def get_mfcc(self, x):
        B, H = tuple(x.shape)
        x = x.view(B, H)
        mfcc = self.mfcc(x)
        return mfcc

    def forward(self, x):
        B = tuple(x.shape)[0]
        mfcc = self.get_mfcc(x)
        # print(mfcc.shape)
        B, H = tuple(x.shape)
        x = x.view(B, 1, -1)
        x = self.sb1(x)
        x = self.sb2(x)
        x = self.sb3(x)
        x = self.sb4(x)
        x = self.sb5(x)
        # x = self.sb6(x)
        # x = self.sb7(x)
        x = x.view(B,128, -1)
        mfcc = mfcc[:,:,:65]
        loss = F.mse_loss(x, mfcc)
        return loss, x, mfcc


class YDataset(Dataset):

    def __init__(self, y=None, frame_len=2 * 8000):
        self.size = len(y)
        self.frame_len = frame_len
        self.y = torch.from_numpy(y)
        # print(self.y.shape)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < (self.size - self.frame_len):
            data = self.y[idx:idx+self.frame_len]
        else:
            data = self.y[idx-self.frame_len:idx]
        return data
