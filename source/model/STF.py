import numpy as np 
import torch
import torchcomplex.nn as nn
from torch.autograd import Variable
import torchcomplex.nn.functional as F
import torch.nn.functional as rF
import random
import copy
import torchvision


ACT_DOMAIN = 'freq'
FILTER_FLAG = True
FREQ_CONV_FLAG = True

#GLOBAL_KERNEL_SIZE = 32

def complex_glorot_uniform(c_in, c_out_total, fft_list, fft_n, use_bias=True, name='complex_mat'):
    c_out = int(c_out_total/len(fft_list))
    kernel = torch.empty((1, 1, int(c_in * c_out), int(fft_n))).cuda()
    kernel = torch.nn.init.xavier_uniform_(kernel)
    kernel_complex_org = torch.fft.fft(torch.complex(kernel, 0.*kernel))
    kernel_complex_org = kernel_complex_org.transpose(1, 2)
    kernel_complex_org = kernel_complex_org[:,:, :, :int(fft_n/2)+1]
    kernel_complex_dict = {}
    for fft_elem in fft_list:
        if fft_elem != fft_n:
            transforms = torchvision.transforms.Resize((1, int(fft_elem/2)+1) )
            kernel_complex_r = transforms(kernel_complex_org.real).transpose(1, 3)
            kernel_complex_i = transforms(kernel_complex_org.imag).transpose(1, 3)
            kernel_complex_dict[fft_elem] = torch.complex(kernel_complex_r, kernel_complex_i).reshape((
                1, 1, int(fft_elem/2)+1, int(c_in), int(c_out))).detach()

        elif fft_elem == fft_n:
            kernel_complex_dict[fft_elem] = kernel_complex_org.reshape((
                1, 1, int(fft_elem/2)+1, int(c_in), int(c_out))).detach()
        kernel_complex_dict[fft_elem].requires_grad = True
        #print(kernel_complex_dict[fft_elem].size())

    bias_complex_r = torch.zeros((c_out*len(fft_list)), requires_grad = True).detach()
    bias_complex_i = torch.zeros((c_out*len(fft_list)), requires_grad = True).detach()
    bias_complex = torch.complex(bias_complex_r, bias_complex_i).cuda()
    return kernel_complex_dict, bias_complex


def zero_interp(in_patch, ratio, out_fft_n):
    # patch_atten with shape (BATCH_SIZE, seg_num, ffn/2+1, c_in)
    in_patch = in_patch.unsqueeze(3)
    in_patch_zero = torch.tile(torch.zeros_like(in_patch),
                        [1, 1, 1, ratio-1, 1])
    in_patch = torch.reshape(torch.cat((in_patch, in_patch_zero), 3), 
                (in_patch.size(0), in_patch.size(1), -1, in_patch.size(-1)) )
    return in_patch[:,:,:out_fft_n,:]


def atten_merge(patch, kernel, bias):
    ## patch with shape (BATCH_SIZE, seg_num, ffn/2+1, c_in, ratio)
    ## kernel with shape (1, 1, 1, 1, ratio, ratio)
    ## bias with shpe (ratio)
    patch_atten = torch.sum(patch.unsqueeze(-1) * kernel, dim = 4)
    # patch_atten with shape (BATCH_SIZE, seg_num, ffn/2+1, c_in, ratio)
    patch_atten = torch.abs(patch_atten + bias)
    patch_atten = F.softmax(patch_atten, dim = -1)
    patch_atten = torch.complex(patch_atten, 0*patch_atten)
    return torch.sum(patch*patch_atten, dim = 4)


def complex_merge(merge_ratio):
    kernel_r = torch.zeros((1, 1, 1, 1, merge_ratio, merge_ratio ), requires_grad=True).cuda()
    kernel_i = torch.zeros((1, 1, 1, 1, merge_ratio, merge_ratio ), requires_grad=True).cuda()
    kernel_complex = torch.complex(kernel_r, kernel_i).detach()

    bias_r = torch.zeros((merge_ratio), requires_grad = True).cuda()
    bias_i = torch.zeros((merge_ratio), requires_grad = True).cuda()
    bias_complex = torch.complex(bias_r, bias_i).detach()

    return kernel_complex, bias_complex





class STFlayer(nn.Module):
    def __init__(self, fft_list, f_step_list, kernel_len_list, dilation_len_list, c_in, c_out,
                         out_fft_list=[0], ser_size=0, pooling=False, BATCH_SIZE = 64):
        super(STFlayer, self).__init__()
        if pooling:
            assert len(fft_list) == len(out_fft_list)
            self.fft_n_list = out_fft_list
        else:
            self.fft_n_list = fft_list
        self.fft_list = fft_list
        self.pooling = pooling
        self.kernel_len_list = kernel_len_list
        self.dilation_len_list = dilation_len_list
        self.c_in = c_in
        self.c_out = c_out
        self.BATCH_SIZE = BATCH_SIZE

        KERNEL_FFT = fft_list[1]
        BASIC_LEN = kernel_len_list[0]
        self.FFT_L_SIZE = len(self.fft_n_list)

        self.patch_kernel_dict, self.patch_bias = complex_glorot_uniform(c_in, c_out, self.fft_n_list, 
                                        KERNEL_FFT)

        self.layerNorms = nn.ModuleList([nn.BatchNorm2d(BATCH_SIZE) for i in range(len(self.fft_n_list))])
        self.convLayers = nn.ModuleList([nn.Conv2d(c_in, int(c_out/len(self.fft_n_list)), (kernel_len_list[i], 1), 
            stride = 1, padding = 0, dilation = dilation_len_list[i]) for i in range(len(self.kernel_len_list)) ] )

        self.merge_kernel = []
        self.merge_bias = []

        for fft_idx, fft_n in enumerate(self.fft_n_list):
            self.merge_kernel.append([])
            self.merge_bias.append([])
            for fft_idx2, tar_fft_n in enumerate(self.fft_n_list):
                if tar_fft_n >= fft_n:
                    time_ratio = int(tar_fft_n / fft_n)
                    kernel, bias = complex_merge(time_ratio)
                    self.merge_kernel[fft_idx].append(kernel)
                    self.merge_bias[fft_idx].append(bias)
                else:
                    self.merge_kernel[fft_idx].append(0)
                    self.merge_bias[fft_idx].append(0)


    def forward(self, inputs):
        # inputs with shape (batch,  c_in, time_len)
        if (inputs.size(0)!=self.BATCH_SIZE):
            print("err")
            pass
        patch_fft_list = []
        patch_mask_list = []
        for idx in range(len(self.fft_n_list)):
            patch_fft_list.append(0.)
            patch_mask_list.append([])

        inputs = inputs.reshape((inputs.size(0) * inputs.size(1), -1))
        #print(inputs.size())
        for fft_idx, fft_n in enumerate(self.fft_n_list):
            # patch_fft with shape (batch, c_in, seg_num, fft_n//2+1)
            if self.pooling:
                in_f_step = self.fft_list[fft_idx]
            else:
                in_f_step = fft_n
            #f_step = in_f_step
            patch_fft = torch.stft(inputs, n_fft = in_f_step, hop_length = in_f_step, return_complex = True, onesided = False).transpose(1, 2)
            patch_fft = patch_fft.reshape(self.BATCH_SIZE, -1, patch_fft.size(-2), patch_fft.size(-1))
            patch_fft = patch_fft[:,:,:,:int(fft_n/2)+1]
            
            patch_fft = patch_fft[:,:,:-1,:]
            patch_fft = patch_fft.permute(0, 2, 3, 1)

            ## patch_fft with shape (batch, seg_num, fft_n//2+1, c_in)
            for fft_idx2, tar_fft_n in enumerate(self.fft_n_list):
                if tar_fft_n < fft_n:
                    continue
                elif tar_fft_n == fft_n:
                    patch_mask = torch.ones_like(patch_fft)
                    for exist_mask in patch_mask_list[fft_idx2]:
                        patch_mask = patch_mask - exist_mask
                    patch_fft_list[fft_idx2] = patch_fft_list[fft_idx2] + patch_mask * patch_fft
                else:                    
                    time_ratio = int(tar_fft_n / fft_n)
                    patch_fft_mod = torch.reshape(patch_fft, 
                        (patch_fft.size(0), -1, time_ratio, patch_fft.size(-2), patch_fft.size(-1) ) )
                    
                    patch_fft_mod = patch_fft_mod.permute(0, 1, 3, 4, 2)
                    patch_fft_mod = atten_merge(patch_fft_mod, self.merge_kernel[fft_idx][fft_idx2], 
                        self.merge_bias[fft_idx][fft_idx2]) * float(time_ratio)
                    
                    patch_mask = torch.ones_like(patch_fft_mod)
                    patch_mask = zero_interp(patch_mask, time_ratio, int(tar_fft_n/2)+1)

                    for exist_mask in patch_mask_list[fft_idx2]:
                        patch_mask = patch_mask - exist_mask

                    patch_mask_list[fft_idx2].append(patch_mask)
                    patch_fft_mod = zero_interp(patch_fft_mod, time_ratio, int(tar_fft_n/2)+1)
                    patch_fft_list[fft_idx2] = patch_fft_list[fft_idx2] + patch_mask * patch_fft_mod
        
        patch_time_list = []
        for fft_idx, fft_n in enumerate(self.fft_n_list):
            # f_step = f_step_list[fft_idx]
            k_len = self.kernel_len_list[fft_idx]
            d_len = self.dilation_len_list[fft_idx]
            paddings = [int((k_len*d_len-d_len)/2), int((k_len*d_len-d_len)/2)]

            patch_fft = patch_fft_list[fft_idx]
            # patch_fft with shape (batch, seg_num, fft_n//2+1, c_in)
            patch_fft = patch_fft.permute(3, 0, 1, 2)#.reshape(patch_fft.size(0), -1, patch_fft.size(-1))
            # patch_fft with shape (c_in, batch, seg_num, fft_n//2+1)
            patch_fft = self.layerNorms[fft_idx](patch_fft)
            patch_fft = patch_fft.permute(1, 2, 3, 0)
            ## patch_fft with shape (batch, seg_num, fft_n//2+1, c_in)
            patch_fft_r, patch_fft_i = patch_fft.real, patch_fft.imag

            if FREQ_CONV_FLAG:
                ## spectral padding
                real_pad_l = torch.flip(patch_fft_r[:,:,1:1+paddings[0],:], [2])
                real_pad_r = torch.flip(patch_fft_r[:,:,-1-paddings[1]:-1,:], [2])
                patch_fft_r = torch.cat((real_pad_l, patch_fft_r, real_pad_r), 2)

                imag_pad_l = torch.flip(patch_fft_i[:,:,1:1+paddings[0],:], [2])
                imag_pad_r = torch.flip(patch_fft_i[:,:,-1-paddings[1]:-1,:], [2])
                patch_fft_i = torch.cat((-imag_pad_l, patch_fft_i, -imag_pad_r), 2)

                patch_fft = torch.complex(patch_fft_r, patch_fft_i)
                #print(patch_fft.size())
                patch_fft = patch_fft.transpose(1, 3)
                # patch_fft with shape (batch, , fft_n//2+1, c_in)
                patch_fft = self.convLayers[fft_idx](patch_fft)
                patch_fft = patch_fft.transpose(1, 3)

            if FILTER_FLAG:
                patch_kernel = self.patch_kernel_dict[fft_n]
                patch_fft = torch.complex(patch_fft_r, patch_fft_i)
                patch_fft = torch.tile(patch_fft.unsqueeze(4), (1, 1, 1, 1, int(self.c_out/self.FFT_L_SIZE)) ) 


                patch_fft_out = patch_fft * patch_kernel
                patch_fft = torch.sum(patch_fft_out, 3)

            patch_out = patch_fft
            if ACT_DOMAIN == 'freq':
                patch_out = F.crelu(patch_out)

            # patch_fft with shape (batch, seg_num, fft_n//2+1, c_out)
            patch_fft_fin = patch_out.permute((0, 3, 2, 1)).reshape(patch_out.size(0) * patch_out.size(3), -1, patch_out.size(1))
            patch_fft_fin = torch.cat((patch_fft_fin, patch_fft_fin[:, :, 0].unsqueeze(-1) ), -1)
            patch_time = torch.istft(patch_fft_fin, n_fft = fft_n, hop_length = fft_n)

            patch_time = patch_time.reshape(self.BATCH_SIZE, -1, patch_time.size(-1)).transpose(1, 2)
            #print(patch_time.size())
            patch_time_list.append(patch_time)

        patch_time_final = torch.cat(patch_time_list, 2)

        if FILTER_FLAG:
            patch_time_final = patch_time_final + self.patch_bias.real

        if ACT_DOMAIN == 'time':
            patch_time_final = rF.relu(patch_time_final)

        return patch_time_final.transpose(1, 2)



class STFNet(nn.Module):
    def __init__(self, args):
        super(STFNet, self).__init__()
        GEN_FFT_N = args.GEN_FFT_N
        GEN_FFT_N2 = args.GEN_FFT_N2
        GEN_FFT_STEP = args.GEN_FFT_STEP
        FILTER_LEN = args.FILTER_LEN
        DILATION_LEN = args.DILATION_LEN
        SENSOR_AXIS = args.SENSOR_AXIS
        SERIES_SIZE = args.input_size
        SERIES_SIZE2 = int(0.75 * args.input_size)
        GEN_C_OUT = 72
        OUT_DIM = args.num_labels
        SENSOR_TYPE = args.sensor_type

        self.SENSOR_TYPE = SENSOR_TYPE
        self.SENSOR_AXIS = SENSOR_AXIS
        assert GEN_C_OUT % SENSOR_TYPE == 0

        self.layers1 = nn.ModuleList([])
        self.layers2 = nn.ModuleList([])
        self.layers3 = nn.ModuleList([])
        self.drop_layers1 = nn.ModuleList([])
        self.drop_layers2 = nn.ModuleList([])
        self.drop_layers3 = nn.ModuleList([])
        for i in range(SENSOR_TYPE):
            self.layers1.append(STFlayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN,
                         int(args.input_channel / args.sensor_type), GEN_C_OUT, ser_size=SERIES_SIZE, BATCH_SIZE = args.batch_size))
            self.layers2.append(STFlayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN,
                         GEN_C_OUT, GEN_C_OUT, ser_size=SERIES_SIZE, BATCH_SIZE = args.batch_size))
            self.layers3.append(STFlayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN,
                         GEN_C_OUT, int(GEN_C_OUT / SENSOR_TYPE), ser_size=SERIES_SIZE, BATCH_SIZE = args.batch_size))
            self.drop_layers1.append(torch.nn.Dropout(p = 0.2))
            self.drop_layers2.append(torch.nn.Dropout(p = 0.2))
            self.drop_layers3.append(torch.nn.Dropout(p = 0.2))


        self.sensor_layer1 = STFlayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
                    int((GEN_C_OUT/2)/len(GEN_FFT_N))*len(GEN_FFT_N)*2, GEN_C_OUT, 
                    out_fft_list=GEN_FFT_N2, ser_size=SERIES_SIZE2, pooling=True , BATCH_SIZE = args.batch_size)

        self.sensor_layer2 = STFlayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
                    GEN_C_OUT, GEN_C_OUT, ser_size=SERIES_SIZE2, BATCH_SIZE = args.batch_size)

        self.sensor_layer3 = STFlayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
                    GEN_C_OUT, GEN_C_OUT, ser_size=SERIES_SIZE2, BATCH_SIZE = args.batch_size)

        self.sensor_drop1 = torch.nn.Dropout(p = 0.2)
        self.sensor_drop2 = torch.nn.Dropout(p = 0.2)
        self.sensor_drop3 = torch.nn.Dropout(p = 0.2)
        self.linear = torch.nn.Linear(GEN_C_OUT, OUT_DIM)

        
    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        BATCH_SIZE = inputs.size(0)
        #inputs: BATCH_SIZE * 9(acc*3, gyro*3, mag*3) * 5 * L
        inputs = torch.reshape(inputs, (BATCH_SIZE, -1, self.SENSOR_TYPE, self.SENSOR_AXIS, inputs.size(-1)) )
        splits = list(torch.split(inputs, 1, 2))
        for i in range(len(splits)):
            splits[i] = torch.reshape(splits[i], (BATCH_SIZE, -1, splits[i].size(-1)) )
            #print(splits[i].size())
            splits[i] = self.drop_layers1[i]( self.layers1[i]( splits[i]))
            splits[i] = self.drop_layers2[i]( self.layers2[i]( splits[i]))
            splits[i] = self.drop_layers3[i]( self.layers3[i]( splits[i]))

        out = torch.cat(splits, 1)

        out = self.sensor_drop1( self.sensor_layer1(out))
        out = self.sensor_drop2( self.sensor_layer2(out))
        out = self.sensor_drop3( self.sensor_layer3(out))

        out = torch.mean(out, -1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    model = STFNet().cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    i = torch.zeros((2, 6, 512), requires_grad = True).cuda()
    p = model(i)
    print(p.size())