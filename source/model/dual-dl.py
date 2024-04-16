#------------
# Author: Shuya Ding
# Date: Sep 2020
#------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.nn.utils import weight_norm
from torchsummary import summary

cnn = models.resnet18(weights=None)
cnn = torch.nn.Sequential(*(list(cnn.children())[:-1]))
output_dim = 512

def block(name, in_feat, out_feat, win_len):    
#     if name == 'IN':
#         layers = [nn.Linear(in_feat, out_feat)]
#         layers.append(nn.InstanceNorm1d(cfg.win_len, 0.8))
#     elif name == 'SN':
#         layers = [SpectralNorm(nn.Linear(in_feat, out_feat))]
        
    if name == 'BN':
        # INPUT: N*win_len*in_feat
        layers = [nn.Linear(in_feat, out_feat)]
        layers.append(nn.BatchNorm1d(win_len, 0.8))
        
#     elif name == 'ADAIN':
#         layers = [nn.Linear(in_feat, out_feat)]
#         layers.append(AdaptiveInstanceNorm1d(cfg.win_len))            
    else:
        layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers    
    
    
class RFNet(nn.Module):
    def __init__(self, num_classes, input_channel, win_len, model = cnn):
        super(RFNet, self).__init__()
        self.fuse = Fusion()
        self.model = model   
        self.hid_dim = 512

        self.fc = nn.Sequential(*block('BN', input_channel * 2, self.hid_dim, win_len))  
        self.lstm_time = nn.LSTM(input_channel, self.hid_dim//2)
        self.lstm_freq = nn.LSTM(input_channel, self.hid_dim//2)    

            
 
        self.classifier = nn.Linear(output_dim, num_classes)
        self.attention = weight_norm(BiAttention(
            time_features=self.hid_dim//2,
            freq_features=self.hid_dim//2,
            mid_features=self.hid_dim,
            glimpses=1,
            drop=0.5,), name='h_weight', dim=None)

        self.apply_attention = ApplyAttention(
            time_features=self.hid_dim//2,
            freq_features=self.hid_dim//2,
            mid_features=self.hid_dim//2,
            glimpses=1,
            num_obj=512,
            drop=0.2,
        )        
        self.cnn1 = torch.nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1)             
        self.fc1 = FCNet(self.hid_dim//2, self.hid_dim//2, 'relu', 0.4)  
        self.fc2 = FCNet(self.hid_dim//2, self.hid_dim//2, 'relu', 0.4)          
        self.fc3 = FCNet(self.hid_dim, self.hid_dim, drop = 0.4)          
    def forward(self,x):
        """
        x: sample_size * 512 * 60
        """
        
        # Time:  sample_size * 512 * 60
        bs,  win_len, dim = x.shape
        
        
        # Freq:  sample_size * 512 * 60
        x_freq = torch.fft.fft(x.permute(0,2,1).reshape(-1,win_len), dim = -1) 
        x_real_freq = x_freq.real.reshape(bs,dim,win_len).permute(0,2,1)
        x_img_freq = x_freq.imag.reshape(bs,dim,win_len).permute(0,2,1)
        x_absolute = torch.sqrt((x_real_freq**2) + (x_img_freq**2)) # sample_size * 512 * 60
        
        
        del x_img_freq, x_real_freq, x_freq 
        torch.cuda.empty_cache()
        
        
        # Cat + FC
        combined = torch.cat([x,x_absolute],-1) # sample_size * 512 * （60*2）
        combined = self.fc(combined) # sample_size * 512 * hid
        
   
        
        
        # CNN Compute
        heat_map = combined.view(bs,win_len,self.hid_dim//2,2).permute(0,3,2,1)
        heat_map = self.cnn1(heat_map)
        feat = self.model(heat_map).squeeze(-1).squeeze(-1) 

        del heat_map, combined
        torch.cuda.empty_cache()



        # Involve attention 
        time = self.lstm_time(x)[0] #bs, win_len, self.hid_dim // 2 
        freq = self.lstm_freq(x_absolute)[0]
        
        del x, x_absolute 
        torch.cuda.empty_cache()             
        
        atten, logits = self.attention(time, freq) 
        time, freq = self.apply_attention(time, freq, atten, logits)

        
        # Time-Tube
        x = self.fc1(time[:,-1,:])
            
        # Freq-Tube
        x_absolute = self.fc2(freq[:,-1,:])        

        del freq,time
        torch.cuda.empty_cache()
        
        feat = self.fc3(torch.cat([x,x_absolute],-1)) + feat
        # Classifier Outputs
        pred = self.classifier(feat)
        
        # del x, x_absolute
        torch.cuda.empty_cache()
        
        return pred#, [x,x_absolute,feat]
    
    
class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)

        self.drop_value = drop
        self.drop = nn.Dropout(drop)

        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None 
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()


    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        
        x = self.lin(x)
        
        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)
    
    
class BiAttention(nn.Module):
    def __init__(self, time_features, freq_features, mid_features, glimpses, drop=0.0):
        super(BiAttention, self).__init__()
        self.hidden_aug = 3
        self.glimpses = glimpses
        self.lin_time = FCNet(time_features, int(mid_features * self.hidden_aug), activate='relu', drop=drop/2.5)  # let self.lin take care of bias
        self.lin_freq = FCNet(freq_features, int(mid_features * self.hidden_aug), activate='relu', drop=drop/2.5)
        
        self.h_weight = nn.Parameter(torch.Tensor(1, glimpses, 1, int(mid_features * self.hidden_aug)).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, glimpses, 1, 1).normal_())

        self.drop = nn.Dropout(drop)

    def forward(self, time, freq):
        """
        time = batch, time_num, dim
        freq = batch, freq_num, dim
        """
        time_num = time.size(1)
        freq_num = freq.size(1)

        time_ = self.lin_time(time).unsqueeze(1)  # batch, 1, time_num, dim
        freq_ = self.lin_freq(freq).unsqueeze(1)  # batch, 1, q_num, dim
        time_ = self.drop(time_)
        
        del time, freq 
        torch.cuda.empty_cache()
        
        
        h_ = time_ * self.h_weight # broadcast:  batch x glimpses x time_num x dim
        logits = torch.matmul(h_, freq_.transpose(2,3)) # batch x glimpses x time_num x freq_num
        
        del h_, freq_
        torch.cuda.empty_cache()        
        
        logits = logits + self.h_bias
        

        torch.cuda.empty_cache()

        atten = F.softmax(logits.view(-1, self.glimpses, time_num * freq_num), 2)
        return atten.view(-1, self.glimpses, time_num, freq_num), logits


class ApplyAttention(nn.Module):
    def __init__(self, time_features, freq_features, mid_features, glimpses, num_obj, drop=0.0):
        super(ApplyAttention, self).__init__()
        self.glimpses = glimpses
        layers = []
        for g in range(self.glimpses):
            layers.append(ApplySingleAttention(time_features, freq_features, mid_features, num_obj, drop))
        self.glimpse_layers = nn.ModuleList(layers)
    
    def forward(self, time, freq,  atten, logits):
        """
        time = batch, time_num, dim
        freq = batch, freq_num, dim
        atten:  batch x glimpses x time_num x freq_num
        logits:  batch x glimpses x time_num x freq_num
        """
        time_num = time.shape[1]
        freq_num = freq.shape[1]
        for g in range(self.glimpses):
            atten_h_freq, atten_h_time = self.glimpse_layers[g](time, freq,  atten[:,g,:,:], logits[:,g,:,:])
            time = atten_h_time + time 
            freq = atten_h_freq + freq
            del atten_h_time, atten_h_freq
            torch.cuda.empty_cache()
        return time, freq
    
class ApplySingleAttention(nn.Module):
    def __init__(self, time_features, freq_features, mid_features, num_obj, drop=0.0):
        super(ApplySingleAttention, self).__init__()
        self.lin_time = FCNet(time_features, time_features, activate='relu', drop=drop) 
        self.lin_freq = FCNet(freq_features, freq_features, activate='relu', drop=drop) 
            
    def forward(self, time, freq, atten, logits):
        """
        time = batch, time_num, dim
        freq = batch, freq_num , dim
       
        atten:  batch x time_num x freq_num
        logits:  batch x time_num x freq_num
        """
        atten_h_time = self.lin_time((time.permute(0,2,1) @ atten).permute(0,2,1))
        del time
        torch.cuda.empty_cache()        
        atten_h_freq = self.lin_freq((freq.permute(0,2,1) @ atten).permute(0,2,1))
        del freq, atten
        torch.cuda.empty_cache()
        
        return atten_h_time, atten_h_freq


    
def main():
    # [@, T, 1, F]
    input = torch.zeros((16, 256, 121, 90)).cuda()
    num_labels = 6
    input_channel = 121
    input_size = 256

    fc_layer = nn.Linear(90, 1, bias=False)
    
    class Squeeze(nn.Module):
        def __init__(self):
            super(Squeeze, self).__init__()

        def forward(self, x):
            return x.squeeze(-1)
    
    model = nn.Sequential(
        fc_layer.cuda(),
        Squeeze().cuda(),
        RFNet(num_classes = num_labels, input_channel = input_channel, win_len = input_size).cuda()
    )
        
    # model = RFNet(num_classes = num_labels, input_channel = input_channel, win_len = input_size).cuda()
    o = model(input)
    # summary(model, input_size=input.size()[1:])
    print(o.size())

if __name__ == '__main__':
	main()