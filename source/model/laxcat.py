import torch
import numpy as np 
import matplotlib.pyplot as plt  
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math


def dense_interpolation(x, M):
    #x: B * C * L
    u = [0 for i in range(M)]
    for t in range(1, 1 + x.size(2)):
        s = M * t / x.size(2)
        for m in range(1, M + 1):
            w = (1 - abs(s - m) / M)**2
            u[m - 1] += w * (x[:, :, t - 1].unsqueeze(-1))
    return torch.cat(u, -1)


class input_attention_layer(nn.Module):
    def __init__(self, p, j):
        super(input_attention_layer, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(p, 1), requires_grad = True)
        self.bias1 = nn.Parameter(torch.zeros(j), requires_grad = True)
        self.weight2 = nn.Parameter(torch.ones(j, p), requires_grad = True)
        self.bias2 = nn.Parameter(torch.zeros(p), requires_grad = True)

    def forward(self, x):
        #x: B * p * j * l
        l = x.size(3)
        h = [0 for i in range(l)]
        x = x.transpose(1, 3)
        #x: B * l * j * p
        for i in range(l):
            tmp = F.relu(torch.matmul(x[:, i, :, :], self.weight1).squeeze(-1) + self.bias1)
            tmp = F.relu(torch.matmul(tmp, self.weight2) + self.bias2)
            #tmp: B * p
            attn = F.softmax(tmp, -1).unsqueeze(1)
            h[i] = torch.sum(attn * x[:, i, :, :], -1)
            h[i] = h[i].unsqueeze(-1) #unsqueeze for cat
            #B * j
        return torch.cat(h, -1)
        #B * j * l
class temporal_attention_layer(nn.Module):
    def __init__(self, j, l):
        super(temporal_attention_layer, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(l, 1), requires_grad = True)
        self.bias1 = nn.Parameter(torch.zeros(j), requires_grad = True)
        self.weight2 = nn.Parameter(torch.ones(j, l), requires_grad = True)
        self.bias2 = nn.Parameter(torch.zeros(l), requires_grad = True)

    def forward(self, x):
        #x: B * j * l
        tmp = F.relu(torch.matmul(x, self.weight1).squeeze(-1) + self.bias1)
        tmp = F.relu(torch.matmul(tmp, self.weight2) + self.bias2)
        attn = F.softmax(tmp, -1).unsqueeze(1)
        #attn: B * 1 * l
        x = torch.sum(attn * x, -1)
        return x



class LaxCat(nn.Module):
    def __init__(self, input_size, input_channel, num_label, hidden_dim = 32, kernel_size = 64, stride = 16):
        super(LaxCat, self).__init__()
        l = int((input_size - kernel_size) / stride) + 1
        self.Conv1 = nn.ModuleList([
            nn.Conv1d(1, hidden_dim, kernel_size = kernel_size, stride = stride) for _ in range(input_channel)])

        self.variable_attn = input_attention_layer(p = input_channel, j = hidden_dim)
        self.temporal_attn = temporal_attention_layer(j = hidden_dim, l = l)
        self.fc = nn.Linear(hidden_dim, num_label)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = list(x.split(1, 1))
        for i in range(len(x)):
            x[i] = self.Conv1[i](x[i]).unsqueeze(-1)

        x = torch.cat(x, -1).permute(0, 3, 1, 2)
        #x = F.relu(self.Conv1(x)).reshape(B, C, -1, x.size(0))
        x = self.variable_attn(x)
        x = self.temporal_attn(x)
        return self.fc(x)




def main():
    stft_m = LaxCat(input_size = 256, input_channel = 726, num_label = 6).cuda()

    total_params = sum(p.numel() for p in stft_m.parameters())
    print(f'{total_params:,} total parameters.')

    #train_STFT_model(stft_m, window_size = 64, K = 16)
    x = torch.zeros(64, 256, 6).cuda()
    output = stft_m(x)
    print(output.size())


if __name__ == '__main__':
    main()
