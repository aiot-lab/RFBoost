import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import AlexNet
class CNN_GRU(nn.Module):
    # 1d-CNN + GRU
    def __init__(self, input_channel, input_size, num_label, n_gru_hidden_units=128, f_dropout_ratio=0.5, batch_first=True):
        super(CNN_GRU, self).__init__()
        # [@, T, C, F]

        # [@, C, F]

        # self.cnn = ResNet(121, input_channel, num_label)
        # self.cnn.fc = nn.Linear(128, 128)

        # self.cnn = nn.Sequential(

        # nn.Conv1d(input_channel, 32, kernel_size=3, stride=1, padding=1),

        # nn.ReLU(inplace=True),

        # nn.AdaptiveMaxPool1d(64),

        # nn.Flatten(),

        # nn.Linear(32 * 64, 128),

        # nn.ReLU(inplace=True),

        # nn.Dropout(f_dropout_ratio),

        # nn.Linear(128, 128),

        # nn.ReLU(inplace=True),

        # )

        
        # [@, T, 64]​

        # self.dropout2 = nn.Dropout(f_dropout_ratio)

        # [Pytorch] DO NOT USE SOFTMAX HERE!!!​

        # self.dense3 = nn.Linear(n_gru_hidden_units, num_label)

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channel, 16, kernel_size=5, stride=1, padding="same"),
            # do layer norm
            nn.LayerNorm([16, input_size]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.LayerNorm([32, input_size // 2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.LayerNorm([64, input_size // 4]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=2),

            # nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            
            # nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            
            nn.Linear(64 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(f_dropout_ratio),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
        )
        # [N, T, C, F]

        self.cnn_ln = nn.LayerNorm(128)

        # [@, T, 64]
        self.rnn = nn.GRU(input_size=128, hidden_size=n_gru_hidden_units, batch_first=batch_first)
        # self.rnn = nn.LSTM(input_size=256, hidden_size=n_gru_hidden_units//2, num_layers=3, batch_first=batch_first, bidirectional=True, dropout=f_dropout_ratio)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=f_dropout_ratio),
            nn.Linear(n_gru_hidden_units, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=f_dropout_ratio),
            nn.Linear(64, num_label),
        )
        
        # for m in self.modules():
        #     if type(m) == nn.GRU:
        #         # GRU: weight: orthogonal, recurrent_kernel: glorot_uniform, bias: zero_initializer
        #         nn.init.orthogonal_(m.weight_ih_l0)
        #         nn.init.orthogonal_(m.weight_hh_l0)
        #         nn.init.zeros_(m.bias_ih_l0)
        #         nn.init.zeros_(m.bias_hh_l0)
        #     elif type(m) == nn.LSTM:
        #         # LSTM: weight: orthogonal, recurrent_kernel: glorot_uniform, bias: zero_initializer
        #         nn.init.orthogonal_(m.weight_ih_l0)
        #         nn.init.orthogonal_(m.weight_hh_l0)
        #         nn.init.zeros_(m.bias_ih_l0)
        #         nn.init.zeros_(m.bias_hh_l0)
                
        #         # init forget gate bias to 1
        #         nn.init.constant_(m.bias_ih_l0[n_gru_hidden_units:2*n_gru_hidden_units], 1)
        #         nn.init.constant_(m.bias_hh_l0[n_gru_hidden_units:2*n_gru_hidden_units], 1)

        #     elif type(m) == nn.Linear:
        #         # Linear: weight: orthogonal, bias: zero_initializer
        #         nn.init.orthogonal_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif type(m) == nn.Conv1d:
        #         # Conv2d: weight: orthogonal, bias: zero_initializer
        #         nn.init.orthogonal_(m.weight)
        #         nn.init.zeros_(m.bias)

    def forward(self, input):
        # [@, T, C]
        cnn_out_list = [self.cnn(input[:, t, :, :]) for t in range(input.size(1))]
        cnn_out = torch.stack(cnn_out_list, dim=1)
        # layer normalization
        # cnn_out = self.cnn_ln(cnn_out)
        # [@, T, 128]
        out, _ = self.rnn(cnn_out)
        x = out[:, -1, :]
        x = self.classifier(x)

        # x = self.dropout2(x)
        # x = self.dense3(x)

        return x

def main():
    # [@, T, 1, F]
    input = torch.zeros((16, 256, 90, 128)).cuda()
    model = CNN_GRU(input_channel = 90, input_size = 121, num_label = 6, n_gru_hidden_units=128, f_dropout_ratio=0.5).cuda()
    o = model(input)
    summary(model, input_size=input.size()[1:])
    print(o.size())

if __name__ == '__main__':
	main()