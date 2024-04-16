import torch
import torch.nn as nn
import torch.nn.functional as F

class MaDNN(nn.Module):
    def __init__(self, input_size, input_channel, num_label):
        super(MaDNN, self).__init__()
        self.Linear1 = nn.ModuleList([nn.Linear(input_size, 128) for _ in range(input_channel)])
        self.Linear2 = nn.ModuleList([nn.Linear(128, 128) for _ in range(input_channel)])
        self.Linear3 = nn.ModuleList([nn.Linear(128, 128) for _ in range(input_channel)])
        self.Linear4 = nn.ModuleList([nn.Linear(128, num_label) for _ in range(input_channel)])

        self.Linear = nn.Linear(input_channel * num_label, num_label)
    def forward(self, x):
        input_channel = x.size(2)
        x = list(x.split(1, 2))
        for i in range(input_channel):
            x[i] = torch.sigmoid(self.Linear1[i](x[i].squeeze(-1)))
            x[i] = torch.sigmoid(self.Linear2[i](x[i]))
            x[i] = torch.sigmoid(self.Linear3[i](x[i]))
            x[i] = torch.sigmoid(self.Linear4[i](x[i]))
        x = torch.cat(x, dim = -1)
        return self.Linear(x)

class MaCNN(nn.Module):
    def __init__(self, input_size, input_channel, num_label, sensor_num):
        super(MaCNN, self).__init__()
        self.in_channel = int(input_channel / sensor_num)
        self.start_conv = nn.ModuleList([nn.Conv1d(self.in_channel, 128, kernel_size = 3, stride = 1, padding = 1) for _ in range(sensor_num)])
        self.conv1 = nn.ModuleList([nn.Conv1d(128, 128, kernel_size = 3, stride = 1, padding = 1) for _ in range(sensor_num)])
        self.pool1 = nn.ModuleList([nn.AvgPool1d(kernel_size = 2) for _ in range(sensor_num)])
        self.conv2 = nn.ModuleList([nn.Conv1d(128, 128, kernel_size = 3, stride = 1, padding = 1) for _ in range(sensor_num)])
        self.pool2 = nn.ModuleList([nn.AvgPool1d(kernel_size = 2)for _ in range(sensor_num)])
        self.conv3 = nn.ModuleList([nn.Conv1d(128, 128, kernel_size = 3, stride = 1, padding = 1) for _ in range(sensor_num)])
        self.pool3 = nn.ModuleList([nn.AvgPool1d(kernel_size = 2)for _ in range(sensor_num)])
        self.conv4 = nn.ModuleList([nn.Conv1d(128, 128, kernel_size = 3, stride = 1, padding = 1) for _ in range(sensor_num)])
        self.pool4 = nn.ModuleList([nn.AvgPool1d(kernel_size = 2)for _ in range(sensor_num)])
        self.conv5 = nn.ModuleList([nn.Conv1d(128, 128, kernel_size = 3, stride = 1, padding = 1) for _ in range(sensor_num)])
        self.pool5 = nn.ModuleList([nn.AvgPool1d(kernel_size = 2)for _ in range(sensor_num)])


        self.end_conv = nn.ModuleList([nn.Conv1d(128, 1, kernel_size = 1, stride = 1) for _ in range(sensor_num)])
        self.Linear = nn.Linear(int(input_size / 32) * sensor_num, num_label)


    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1, self.in_channel).transpose(1, 3)
        sensor_num = x.size(2)
        x = list(x.split(1, 2))
        for i in range(sensor_num):
            x[i] = F.relu(self.start_conv[i](x[i].squeeze(2)))

            x[i] = F.relu(self.pool1[i](self.conv1[i](x[i])))
            x[i] = F.relu(self.pool2[i](self.conv2[i](x[i])))
            x[i] = F.relu(self.pool3[i](self.conv3[i](x[i])))
            x[i] = F.relu(self.pool4[i](self.conv4[i](x[i])))
            x[i] = F.relu(self.pool5[i](self.conv5[i](x[i])))
            x[i] = F.relu(self.end_conv[i](x[i])).squeeze(1)

            x[i] = x[i].view(x[i].size(0), -1)
        x = torch.cat(x, dim = -1)
        return self.Linear(x)


def main():
    input = torch.zeros((4, 256, 726)).cuda()
    model = MaCNN(input_size = 256, input_channel = 726, num_label = 6, sensor_num = 6).cuda()
    o = model(input)
    print(o.size())


    
if __name__ == '__main__':
    main()
