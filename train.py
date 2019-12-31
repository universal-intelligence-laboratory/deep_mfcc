import glob
import os

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
import random
from tensorboardX import SummaryWriter

from model import YDataset, SampleMel, MelnnAudio
from download import maybe_download_and_extract_dataset

writer = SummaryWriter(log_dir='./MelnnAudio_xmask_NormNone_trainable/')

def train():
    data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    data_dir = "./speech_dataset/"
    maybe_download_and_extract_dataset(data_url, data_dir)

    device = "cpu"
    data_path = "y_all.npy"
    if not os.path.exists(data_path):
        y_all = []
        for i in tqdm.tqdm(random.choices(glob.glob(os.path.join(data_dir, "*/*.wav")), k = 1000)):
            y, sr = librosa.load(i, sr=None, mono=True)
            y_all.extend(list(y))

        y_all = np.array(y_all)
        np.save(data_path, y_all)
    else:
        y_all = np.load(data_path)

    print(y_all.shape)
    train_loader = DataLoader(YDataset(y_all), batch_size=256, shuffle=True)

    model = MelnnAudio().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    model.train()

    epoch = 1000
    for i in range(epoch):
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for batch_idx, y in enumerate(train_loader):
                global_step = i * epoch + batch_idx
                optimizer.zero_grad()
                loss, x, mfcc = model(y.to(device))
                loss.backward()
                optimizer.step()
                pbar.update(1)
                
                if batch_idx % 100 == 0:
                    pseudo_mfcc = np.expand_dims(x.detach().cpu().numpy()[0], 0)
                    real_mfcc = np.expand_dims(mfcc.detach().cpu().numpy()[0], 0)
                    # mask = model.mask
                    # writer.add_histogram('mask', mask, global_step)
                    writer.add_image('pseudo_mfcc', pseudo_mfcc, global_step)
                    writer.add_image('real_mfcc', real_mfcc, global_step)
                    writer.add_scalar('loss', loss.detach().cpu().numpy(), global_step)
                    


if __name__ == "__main__":
    train()
