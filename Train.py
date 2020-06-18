import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy.io.wavfile
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np
from librosa.core import load
import librosa as librosa
import librosa.display as d
import matplotlib.pyplot as plt
from librosa.util import normalize
from NN.Generator import Generator
from NN.Discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter

# Hyperprameters
lr = 0.0002
epochs = 100
N_disc_blocks = 3
N_disc_layers = 4
λ = 10

filename = "dataset\\000-elevator.wav"

y, sr = load(filename, sr=44100)

# Initieal mel-spectogram
x = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=1024, win_length=1024, hop_length=256)


x = torch.from_numpy(x).float()

x = x.unsqueeze(0)


# Create discriminator and generator
g = Generator(256, 64).cuda()
d = Discriminator(128).cuda()

# Setup Optimizer for G and D
optG = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
optD = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))


for epoch in range(epochs):
    new_x = g(x.cuda())
    # Train Discriminator
    d.zero_grad()

    D_fake = d(new_x.cuda().detach())
    D_real = d(x.cuda())

    D_total_loss = 0

    for scale in D_fake:
        D_total_loss += F.relu(1 + scale[-1]).mean()
        # print(F.relu(1 + scale[-1]).mean())

    for scale in D_real:
        D_total_loss += F.relu(1 - scale[-1]).mean()
        # print((1 - scale[-1]).mean())

    D_total_loss.backward()
    optD.step()

    # Train Generator
    g.zero_grad()

    D_fake_new = d(new_x.cuda())

    G_loss = 0
    for scale in D_fake_new:
        G_loss += -scale[-1].mean()

    feature_loss = 0
    weight = 1 / N_disc_layers

    for i in range(N_disc_blocks):
        for j in range(len(D_fake[i]) - 1):
            feature_loss += weight * \
                F.l1_loss(D_fake_new[i][j], D_real[i][j].detach(),)

    (G_loss + λ * feature_loss).backward()
    optG.step()

    print("Epoch: %d" % epoch + " D_loss:%4f" % D_total_loss + "   G_loss:%f" %
          (G_loss + λ * feature_loss))


S = new_x.squeeze().detach().cpu().numpy()




N = librosa.feature.inverse.mel_to_audio(
    S, 44100, n_fft=1024, win_length=1024, hop_length=256, pad_mode='reflect')



scipy.io.wavfile.write("dataset/g.wav", 44100, N)
