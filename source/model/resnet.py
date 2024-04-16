import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math

class resConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, layer_num):
        super(resConv1dBlock, self).__init__()
        self.layer_num = layer_num
        self.conv1 = nn.ModuleList([
            nn.Conv1d(in_channels = in_channels, out_channels = 2 * in_channels, kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) / 2) )
            for i in range(layer_num)])

        self.bn1 = nn.ModuleList([
            nn.BatchNorm1d(2 * in_channels)
            for i in range(layer_num)])

        self.conv2 = nn.ModuleList([ 
            nn.Conv1d(in_channels = 2 * in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) / 2) )
            for i in range(layer_num)])

        self.bn2 = nn.ModuleList([
            nn.BatchNorm1d(out_channels)
            for i in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            tmp = F.relu(self.bn1[i](self.conv1[i](x)))
            x = F.relu(self.bn2[i](self.conv2[i](tmp)) + x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_size, input_channel, num_label):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size = 1, stride = 1)
        self.res1 = resConv1dBlock(64, 64, kernel_size = 3, stride = 1, layer_num = 3)
        self.pool1 = nn.AvgPool1d(kernel_size = 2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size = 1, stride = 1)
        self.res2 = resConv1dBlock(128, 128, kernel_size = 3, stride = 1, layer_num = 4)
        self.pool2 = nn.AvgPool1d(kernel_size = 2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size = 1, stride = 1)
        self.res3 = resConv1dBlock(256, 256,  kernel_size = 3, stride = 1, layer_num = 7)
        self.pool3 = nn.AvgPool1d(kernel_size = 2)

        self.conv4 = nn.Conv1d(256, 128, kernel_size = 1, stride = 1)
        self.res4 = resConv1dBlock(128, 128, kernel_size = 3, stride = 1, layer_num = 4)
        self.pool = nn.AvgPool1d(kernel_size = int(input_size / 8))

        self.fc = nn.Linear(128, num_label)

    def forward(self, x, til_layer="final"):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool1(self.res1(x))
        if til_layer == 'res1':
            return x
        x = F.relu(self.conv2(x))
        x = self.pool2(self.res2(x))
        if til_layer == 'res2':
            return x
        x = F.relu(self.conv3(x))
        x = self.pool3(self.res3(x))
        if til_layer == 'res3':
            return x
        x = F.relu(self.conv4(x))
        x = self.pool(self.res4(x))
        if til_layer == 'res4':
            return x
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main():
    input = torch.zeros((4, 256, 45)).cuda()
    model = ResNet(input_size = 256, input_channel = 45, num_label = 6).cuda()
    o = model(input)
    print(o.size())

if __name__ == '__main__':
	main()