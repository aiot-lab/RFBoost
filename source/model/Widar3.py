import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import AlexNet

class Widar3(nn.Module):
    # CNN + GRU
    def __init__(self, input_shape, input_channel, num_label, n_gru_hidden_units=128, f_dropout_ratio=0.5, batch_first=True):
        super(Widar3, self).__init__()
        self.num_label = num_label
        self.n_gru_hidden_units = n_gru_hidden_units
        self.f_dropout_ratio = f_dropout_ratio
        # [@, T_MAX, 1, 20, 20] 
        self.input_shape = input_shape
        self.input_time = input_shape[1]
        self.input_channel = input_shape[2]
        self.input_x, self.input_y = input_shape[3], input_shape[4]

        # self.cnn = AlexNet(weights=None).features
        # self.cnn.

        self.Tconv1_out_channel=16
        self.Tconv1_kernel_size=5
        self.Tconv1_stride=1

        self.Tdense1_out = 64
        self.Tdense2_out = 64
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, self.Tconv1_out_channel, self.Tconv1_kernel_size, self.Tconv1_stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Flatten(),
            nn.Linear(self.Tconv1_out_channel * ((self.input_x - 4) // 2) * ((self.input_y - 4) // 2), self.Tdense1_out),
            nn.ReLU(inplace=True),
            nn.Dropout(f_dropout_ratio),
            nn.Linear(self.Tdense1_out, self.Tdense2_out),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=self.Tdense2_out, hidden_size=n_gru_hidden_units, batch_first=batch_first)
        self.dropout2 = nn.Dropout(f_dropout_ratio)
        # [Pytorch] DO NOT USE SOFTMAX HERE!!!
        self.dense3 = nn.Linear(n_gru_hidden_units, num_label)
        
    def forward(self, input):
        # [@, T_MAX, 1, 20, 20]
        cnn_out_list = [self.cnn(input[:, t, :, :, :]) for t in range(self.input_time)]
        cnn_out = torch.stack(cnn_out_list, dim=1)
        # [@, T_MAX, 64]
        out, _ = self.gru(cnn_out)
        x = out[:, -1, :]
        x = self.dropout2(x)
        x = self.dense3(x)
        # x = F.relu(x)
        return x

def main():
    input = torch.zeros((4, 38, 1, 20, 20)).cuda()
    model = Widar3(input.shape, input_channel = 1, num_label = 6, n_gru_hidden_units=128, f_dropout_ratio=0.5).cuda()
    o = model(input)
    summary(model, input_size=input.size()[1:])
    print(o.size())

if __name__ == '__main__':
	main()

# import torch
# import numpy as np 
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# # Tensorflow
# # model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')    # (@,T_MAX,20,20,1)
# # # Feature extraction part
# # x = TimeDistributed(Conv2D(16,kernel_size=(5,5),activation='relu',data_format='channels_last',\
# #     input_shape=input_shape))(model_input)   # (@,T_MAX,20,20,1)=>(@,T_MAX,16,16,16)
# # x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)    # (@,T_MAX,16,16,16)=>(@,T_MAX,8,8,16)
# # x = TimeDistributed(Flatten())(x)   # (@,T_MAX,8,8,16)=>(@,T_MAX,8*8*16)
# # x = TimeDistributed(Dense(64,activation='relu'))(x) # (@,T_MAX,8*8*16)=>(@,T_MAX,64)
# # x = TimeDistributed(Dropout(f_dropout_ratio))(x)
# # x = TimeDistributed(Dense(64,activation='relu'))(x) # (@,T_MAX,64)=>(@,T_MAX,64)
# # x = GRU(n_gru_hidden_units,return_sequences=False)(x)  # (@,T_MAX,64)=>(@,128)
# # x = Dropout(f_dropout_ratio)(x)
# # model_output = Dense(n_class, activation='softmax', name='name_model_output')(x)  # (@,128)=>(@,n_class)

# # # Model compiling
# # model = Model(inputs=model_input, outputs=model_output)
# # model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
# #                 loss='categorical_crossentropy',
# #                 metrics=['accuracy']
# #             )

# # Pytorch
# # class TimeDistributed(nn.Module):
# #     def __init__(self, module, batch_first=True):
# #         super(TimeDistributed, self).__init__()
# #         self.module = module
# #         self.batch_first = batch_first

# #     def forward(self, x):
# #         if len(x.size()) <= 2:
# #             return self.module(x)
# #         # Input: (batch, time, channels, x, y)
        
# #         # split time dimension
# #         x_split = x.split(1, dim=1)
# #         x_split = [e.squeeze(1) for e in x_split]
# #         # Apply the module on each timestep
# #         outputs = [self.module(x_t) for x_t in x_split]
# #         # # Combine the output tensors back into a single tensor
# #         return torch.stack(outputs, dim=1) if self.batch_first else torch.stack(outputs, dim=2)

# class TimeDistributed(nn.Module):
#     def __init__(self, module, batch_first=True):
#         super(TimeDistributed, self).__init__()
#         self.module = module
#         self.batch_first = batch_first

#     def forward(self, x):
#         assert len(x.size()) > 2

#         # reshape input data --> (samples * timesteps, input_size)
#         # squash timesteps
#         batch_size, time_steps, C, H, W = x.size()
#         c_in = x.view(batch_size * time_steps, C, H, W)
#         c_out = self.module(c_in)
#         r_in = c_out.view(batch_size, time_steps, -1)
#         if self.batch_first is False:
#             r_in = r_in.permute(1, 0, 2)
#         return r_in


# class Widar3(nn.Module):
#     def __init__(self, input_shape, input_channel, num_label, n_gru_hidden_units=128, f_dropout_ratio=0.5, batch_first=True):
#         super(Widar3, self).__init__()
#         self.num_label = num_label
#         self.n_gru_hidden_units = n_gru_hidden_units
#         self.f_dropout_ratio = f_dropout_ratio
#         # [@, T_MAX, 1, 20, 20] 
#         self.input_shape = input_shape
#         self.input_time = input_shape[1]
#         self.input_channel = input_shape[2]
#         self.input_x, self.input_y = input_shape[3], input_shape[4]

#         self.Tconv1_out_channel=16
#         self.Tconv1_kernel_size=5
#         self.Tconv1_stride=1

#         self.Tdense1_out = 64
#         self.Tdense2_out = 64

#         data_shape = input_shape
#         self.Tconv1 = TimeDistributed(nn.Conv2d(input_channel, self.Tconv1_out_channel, self.Tconv1_kernel_size, self.Tconv1_stride), batch_first=batch_first)
#         data_shape = [data_shape[0], data_shape[1], self.Tconv1_out_channel, data_shape[3] - 4, data_shape[4] - 4]
#         self.Tmaxpool1 = TimeDistributed(nn.MaxPool2d(kernel_size=(2,2)))
#         data_shape = [data_shape[0], data_shape[1], data_shape[2], data_shape[3] // 2, data_shape[4] // 2]
#         self.Tflatten1 = TimeDistributed(nn.Flatten())
#         data_shape = [data_shape[0], data_shape[1], data_shape[2] * data_shape[3] * data_shape[4]]
#         self.Tdense1 = TimeDistributed(nn.Linear(data_shape[2], self.Tdense1_out))
#         data_shape = [data_shape[0], data_shape[1], self.Tdense1_out]
#         self.Tdropout1 = TimeDistributed(nn.Dropout(f_dropout_ratio))
#         data_shape = [data_shape[0], data_shape[1], data_shape[2]]
#         self.Tdense2 = TimeDistributed(nn.Linear(self.Tdense1_out, self.Tdense2_out))
#         data_shape = [data_shape[0], data_shape[1], self.Tdense2_out]
#         self.gru = nn.GRU(input_size=self.Tdense2_out, hidden_size=n_gru_hidden_units, batch_first=batch_first)

#         self.dropout2 = nn.Dropout(f_dropout_ratio, inplace=False)
#         # self.dense3 = nn.Sequential(nn.Linear(n_gru_hidden_units, num_label), nn.Softmax(dim=1))
#         # DO NOT USE SOFTMAX
#         self.dense3 = nn.Linear(n_gru_hidden_units, num_label)
#         data_shape = [data_shape[0], num_label]


#         # initialize weights
#         for m in self.modules():
#             if type(m) == nn.GRU:
#                 # GRU: weight: orthogonal, recurrent_kernel: glorot_uniform, bias: zero_initializer
#                 nn.init.orthogonal_(m.weight_ih_l0)
#                 nn.init.orthogonal_(m.weight_hh_l0)
#                 nn.init.zeros_(m.bias_ih_l0)
#                 nn.init.zeros_(m.bias_hh_l0)
#             elif type(m) == nn.Linear:
#                 # Linear: weight: orthogonal, bias: zero_initializer
#                 nn.init.orthogonal_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif type(m) == nn.Conv2d:
#                 # Conv2d: weight: orthogonal, bias: zero_initializer
#                 nn.init.orthogonal_(m.weight)
#                 nn.init.zeros_(m.bias)
                    



        
#     def forward(self, input):
#         # [@, T_MAX, 1, 20, 20] -> [@, T_MAX, 16, 16, 16]
#         x = self.Tconv1(input)
#         # Relu
#         x = F.relu(x)
#         # [@, T_MAX, 16, 16, 16] -> [@, T_MAX, 16, 8, 8]
#         x = self.Tmaxpool1(x)
#         # [@, T_MAX, 16, 8, 8] -> [@, T_MAX, 16*8*8]
#         x = self.Tflatten1(x)
#         # [@, T_MAX, 16*8*8] -> [@, T_MAX, 64]
#         x = self.Tdense1(x)
#         # Relu
#         x = F.relu(x)
#         # [@, T_MAX, 64] -> [@, T_MAX, 64]
#         x = self.Tdropout1(x)
#         # [@, T_MAX, 64] -> [@, T_MAX, 64]
#         x = self.Tdense2(x)
#         # Relu
#         x = F.relu(x)
#         # [@, T_MAX, 64] -> [@, 128]
#         # keras.layers.GRU(n_gru_hidden_units,return_sequences=False)(x)
#         x = self.gru(x)[0][:, -1, :]
#         x = x.squeeze(0)
#         # [@, 128] -> [@, 128]
#         x = self.dropout2(x)
#         # [@, 128] -> [@, n_class]
#         x = self.dense3(x)
#         # check any nan
#         if torch.isnan(x).any():
#             print('nan')
#             print(x)
#             exit()
#         return x

def main():
    input = torch.zeros((4, 38, 1, 200, 20)).cuda()
    model = Widar3(input.shape, input_channel = 1, num_label = 6, n_gru_hidden_units=128, f_dropout_ratio=0.5).cuda()
    o = model(input)
    summary(model, input_size=input.size()[1:])
    print(o.size())

# if __name__ == '__main__':
# 	main()