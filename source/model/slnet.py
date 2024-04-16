import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import AlexNet

import os, sys, math
import numpy as np
import scipy.io as scio
import torch, torchvision
import torch.nn as nn
from torch import sigmoid
from torch.fft import fft, ifft
from torch.nn.functional import relu
from torch import sigmoid

class m_Linear(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out

        # Creation
        self.weights_real = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.weights_imag = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(2, size_out, dtype=torch.float32))

        # Initialization
        nn.init.xavier_uniform_(self.weights_real, gain=1)
        nn.init.xavier_uniform_(self.weights_imag, gain=1)
        nn.init.zeros_(self.bias)
    
    def swap_real_imag(self, x):
        # [@,*,2,Hout]
        # [real, imag] => [-1*imag, real]
        h = x                   # [@,*,2,Hout]
        h = h.flip(dims=[-2])   # [@,*,2,Hout]  [real, imag]=>[imag, real]
        h = h.transpose(-2,-1)  # [@,*,Hout,2]
        h = h * torch.tensor([-1,1]).cuda()     # [@,*,Hout,2] [imag, real]=>[-1*imag, real]
        h = h.transpose(-2,-1)  # [@,*,2,Hout]
        
        return h

    def forward(self, x):
        # x: [@,*,2,Hin]
        # Note: torch.mm function doesn't support broadcasting

        h = x           # [@,*,2,Hin]

        h1 = torch.matmul(h, self.weights_real) # [@,*,2,Hout]
        h2 = torch.matmul(h, self.weights_imag) # [@,*,2,Hout]
        h2 = self.swap_real_imag(h2)            # [@,*,2,Hout]
        h = h1 + h2                             # [@,*,2,Hout]
        h = torch.add(h, self.bias)             # [@,*,2,Hout]+[2,Hout]=>[@,*,2,Hout]

        return h

class m_cconv3d(nn.Module):
    def __init__(self, D_cnt, F_bins, T_bins):
        super().__init__()
        # input:  (@,2,D_cnt,F_bins,T_bins)
        # output: (@,2,D_cnt,F_bins,T_bins)
        # kernel: (D_cnt,F_bins,T_bins)
        # bias:   (D_cnt,1,T_bins)

        # Parameters offloading
        self.D_cnt, self.F_bins, self.T_bins = D_cnt, F_bins, T_bins

        # Parameter initialization
        self.fc_1 = nn.Linear(self.F_bins, self.F_bins)
        self.fc_2 = nn.Linear(self.F_bins, self.F_bins)

        # Meta matrix for DFT/IDFT
        F_range = torch.arange(self.F_bins)
        phase = -2 * math.pi * (F_range * F_range.reshape((self.F_bins,1))) / self.F_bins
        self.DFT_COS_M = torch.cos(phase).cuda()
        self.DFT_SIN_M = torch.sin(phase).cuda()

    def _mul_complex(self, a, b):
        # a: (@,2,D,T,F), b: (@,2,D,T,F)
        h1 = a * b                              # (@,2,D,T,F)
        ret_real = h1[:,0,:] - h1[:,1,:]        # (@,D,T,F)

        h2 = a * b.flip(dims=[1])               # (@,2,D,T,F)
        ret_imag = h2[:,0,:] + h2[:,1,:]        # (@,D,T,F)
        ret = torch.stack((ret_real,ret_imag), axis=1)  # (@,D,T,F)=>(@,2,D,T,F)
        return ret

    def _fft_complex(self, x):
        return self._dft_complex(x)

    def _ifft_complex(self, x):
        return self._idft_complex(x)

    def _dft_complex(self, x):
        h = x           # (@,2,D,T,F)

        ret_real = torch.matmul(h[:,0,:], self.DFT_COS_M) - torch.matmul(h[:,1,:], self.DFT_SIN_M)
        ret_imag = torch.matmul(h[:,0,:], self.DFT_SIN_M) + torch.matmul(h[:,1,:], self.DFT_COS_M)

        ret = torch.stack((ret_real, ret_imag), axis=1)     # (@,D,T,F)=>(@,2,D,T,F)
        return ret

    def _idft_complex(self, x):
        h = x           # (@,2,D,T,F)

        ret_real = torch.matmul(h[:,0,:],      self.DFT_COS_M)/self.F_bins + torch.matmul(h[:,1,:], self.DFT_SIN_M)/self.F_bins
        ret_imag = torch.matmul(h[:,0,:], -1 * self.DFT_SIN_M)/self.F_bins + torch.matmul(h[:,1,:], self.DFT_COS_M)/self.F_bins

        ret = torch.stack((ret_real, ret_imag), axis=1)     # (@,D,T,F)=>(@,2,D,T,F)
        return ret
    
    def _cconv(self, x, y):
        # cconv with 'N*fft(ifft(a)*ifft(b))'
        x_ifft = self._ifft_complex(x)              # (@,2,D,T,F)
        y_ifft = self._ifft_complex(y)              # (@,2,D,T,F)
        h = self._mul_complex(x_ifft, y_ifft)       # (@,2,D,T,F)
        ret = self.F_bins * self._fft_complex(h)    # (@,2,D,T,F)
        return ret

    def forward(self, x):
        h = x           # (@,2,D,F,T)

        h = h.permute(0,1,2,4,3)                # (@,2,D,F,T)=>(@,2,D,T,F)

        w = h                                   # (@,2,D,T,F)
        w = self.fc_1(w)                        # (@,2,D,T,F)
        w = self.fc_2(w)                        # (@,2,D,T,F)
        w = torch.nn.Softmax(dim=-1)(w)
        h2 = self._cconv(h,w)                   # (@,2,D,T,F)

        output = h2.permute(0,1,2,4,3)              # (@,2,D,T,F)=>(@,2,D,F,T)
        return output

class m_Filtering(nn.Module):
    def __init__(self, D_cnt, F_bins, T_bins):
        super().__init__()
        # input:  (@,2,D_cnt,F_bins,T_bins)
        # output: (@,2,D_cnt,F_bins,T_bins)

        # Parameters offloading
        self.D_cnt, self.F_bins, self.T_bins = D_cnt, F_bins, T_bins

        # Parameter initialization
        self.fc_1 = nn.Linear(self.F_bins, self.F_bins)

    def forward(self, x):
        h = x                   # (@,2,D,F,T)

        # Self-attention to generate filter weights
        w = torch.linalg.norm(h,dim=1)          # (@,2,D,F,T)=>(@,D,F,T)
        w = w.permute(0,1,3,2)                  # (@,D,F,T)=>(@,D,T,F)
        w = sigmoid(self.fc_1(w))               # (@,D,T,F)
        w = w.permute(0,1,3,2)                  # (@,D,T,F)=>(@,D,F,T)
        w = w.unsqueeze(dim=1)                  # (@,D,F,T)=>(@,1,D,F,T)

        output = h * w                          # (@,2,D,F,T)
        return output

class m_pconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_front_pconv_layer):
        super().__init__()
        # input:  (@,2,C_in,D,F,T) ~ e.g., (@,2,1,6,121,T)
        # output: (@,2,C_out,D_out,F_out,T_out)

        # Parameters offloading
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_front_pconv_layer = is_front_pconv_layer

        # Conventional Convolution Initialization
        self.conv3d = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,\
            kernel_size=self.kernel_size, stride=self.stride)

    def _polarize(self, h, F):
        # Abandon original phase of F
        # Polarize zone: [-pi/2, pi/2]
        h = h.permute(0,2,3,5,1,4)              # (@,2,C_in,D,F,T)=>(@,C_in,D,T,2,F)
        h = torch.linalg.norm(h,dim=4)          # (@,C_in,D,T,2,F)=>(@,C_in,D,T,F)
        cos_matrix = torch.cos(torch.linspace(-1*math.pi/1, 1*math.pi/1, F))  # (F,)
        sin_matrix = torch.sin(torch.linspace(-1*math.pi/1, 1*math.pi/1, F))  # (F,)

        h_cos = h * cos_matrix.cuda()           # (@,C_in,D,T,F)
        h_sin = h * sin_matrix.cuda()           # (@,C_in,D,T,F)

        h = torch.stack((h_cos,h_sin), axis=1)  # (@,C_in,D,T,F)=>(@,2,C_in,D,T,F)
        
        h_polarized = h.permute(0,1,2,3,5,4)    # (@,2,C_in,D,T,F)=>(@,2,C_in,D,F,T)
        return h_polarized

    def forward(self, x):
        h = x   # (@,2,C_in,D,F,T) ~ e.g., (@,2,1,6,121,T)
        [D,F,T] = x.shape[3:]

        # Polarize for the front pconv layer
        if self.is_front_pconv_layer:
            h = self._polarize(h, F)        # (@,2,C_in,D,F,T)
        
        # Conventional 3D Convolution
        h = h.reshape((-1,self.in_channels,D,F,T))          # (@,2,C_in,D,F,T)=>(@*2,C_in,D,F,T)
        h = self.conv3d(h)                                  # (@*2,C_in,D,F,T)=>(@*2,C_out,D_out,F_out,T_out)
        output = h.reshape((-1,2)+h.shape[1:])    # (@*2,C_out,D_out,F_out,T_out)=>(@,2,C_out,D_out,F_out,T_out)
        return output


   
class SLNet(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(SLNet, self).__init__()
        self.input_shape = input_shape  # [2,R,6,121,T_MAX]
        self.IN_CHANNEL = input_shape[1]
        self.NUM_RX = input_shape[2]
        self.T_MAX = input_shape[4]
        self.class_num = class_num

        # pconv+FC
        self.complex_fc_1 = m_Linear(32*7, 128)
        self.complex_fc_2 = m_Linear(128, 64)
        self.fc_1 = nn.Linear(32*7, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 32)
        self.fc_4 = nn.Linear(self.NUM_RX*(self.T_MAX-8)*32, 256)
        self.fc_5 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, self.class_num)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.dropout_4 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.2)
        self.pconv3d_1 = m_pconv3d(in_channels=self.IN_CHANNEL,out_channels=16,kernel_size=[1,5,5],stride=[1,1,1],is_front_pconv_layer=True)
        self.pconv3d_2 = m_pconv3d(in_channels=16,out_channels=32,kernel_size=[1,5,5],stride=[1,1,1],is_front_pconv_layer=False)
        self.mpooling3d_1 = nn.MaxPool3d(kernel_size=[1,3,1],stride=[1,3,1])
        self.mpooling3d_2 = nn.MaxPool3d(kernel_size=[1,5,1],stride=[1,5,1])

    def forward(self, x):
        h = x   # [@,2,R,C,F,T] ~ (@,2,1,6,121,T_MAX) or (@,2,3,6,121,T_MAX)

        # pconv
        h = self.pconv3d_1(h)                       # (@,2,R,6,121,T_MAX)=>(@,2,16,6,117,T_MAX-4)
        h = h.reshape((-1,16,self.NUM_RX,117,self.T_MAX-4))   # (@,2,16,6,117,T_MAX-4)=>(@*2,16,6,117,T_MAX-4)
        h = self.mpooling3d_1(h)                    # (@*2,16,6,117,T_MAX-4)=>(@*2,16,6,39,T_MAX-4)
        h = h.reshape((-1,2,16,self.NUM_RX,39,self.T_MAX-4))  # (@*2,16,6,39,T_MAX-4)=>(@,2,16,6,39,T_MAX-4)

        h = self.pconv3d_2(h)                       # (@,2,16,6,39,T_MAX-4)=>(@,2,32,6,35,T_MAX-8)
        h = h.reshape((-1,32,self.NUM_RX,35,self.T_MAX-8))    # (@,2,32,6,35,T_MAX-8)=>(@*2,32,6,35,T_MAX-8)
        h = self.mpooling3d_2(h)                    # (@*2,32,6,35,T_MAX-8)=>(@*2,32,6,7,T_MAX-8)
        h = h.reshape((-1,2,32,self.NUM_RX,7,self.T_MAX-8))   # (@*2,32,6,7,T_MAX-8)=>(@,2,32,6,7,T_MAX-8)

        # Complex FC
        h = h.permute(0,3,5,1,2,4)                  # (@,2,32,6,7,T_MAX-8)=>(@,6,T_MAX-8,2,32,7)
        h = h.reshape((-1,self.NUM_RX,self.T_MAX-8,2,32*7))   # (@,6,T_MAX-8,2,32,7)=>(@,6,T_MAX-8,2,32*7)
        h = self.dropout_1(h)
        h = self.complex_fc_1(h)                    # (@,6,T_MAX-8,2,32*7)=>(@,6,T_MAX-8,2,128)
        h = self.dropout_2(h)
        h = self.complex_fc_2(h)                    # (@,6,T_MAX-8,2,128)=>(@,6,T_MAX-8,2,64)

        # FC
        h = torch.linalg.norm(h,dim=3)              # (@,6,T_MAX-8,2,64)=>(@,6,T_MAX-8,64)
        h = relu(self.fc_3(h))                      # (@,6,T_MAX-8,64)=>(@,6,T_MAX-8,32)
        h = h.reshape((-1,self.NUM_RX*(self.T_MAX-8)*32))     # (@,6,T_MAX-8,32)=>(@,6*(T_MAX-8)*32)
        h = self.dropout_3(h)
        h = relu(self.fc_4(h))                      # (@,6*(T_MAX-8)*32)=>(@,256)
        h = self.dropout_4(h)
        h = relu(self.fc_5(h))                      # (@,256)=>(@,128)
        h = self.dropout_5(h)
        output = self.fc_out(h)           # (@,128)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        return output
    
def main():
    # [@, T, 1, F]
    input = torch.zeros((8,2,1,1,121,256)).cuda()
    model = SLNet(input.shape[1:], class_num = 6).cuda()
    o = model(input)
    summary(model, input_size=input.size()[1:])
    print(o.size())

if __name__ == '__main__':
	main()