#Original model presented in: C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah, Deep Learning Human Mind for Automated Visual Classification, CVPR 2017 
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np

class Model(nn.Module):

    def __init__(self, input_size=128, lstm_size=128, lstm_layers=4, output_size=128, include_top=True, n_classes=40):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.include_top = include_top

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.L0 = nn.Linear(lstm_size, output_size)
        # self.L1p = nn.Linear(output_size, output_size)
        # self.L2p = nn.Linear(output_size, output_size)

        self.classifier = nn.Linear(output_size,n_classes)
        
    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size, dtype=torch.float), torch.zeros(self.lstm_layers, batch_size, self.lstm_size, dtype=torch.float))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))

        x = self.lstm(x, lstm_init)[0][:,-1,:]

        # x, _ = self.lstm(x)
        # Get the output from the last time step
        x = self.L0(x)


        # print(lstm_init[0].dtype, x.dtype)

        # Forward LSTM and get final state
        # x = self.lstm(x, lstm_init)[0][:,-1,:]
        # x = F.gelu(x)
        # Forward output
        # x = self.L0(x)
        # x = F.dropout(x, p=0.5)
        # x = F.gelu(self.L1p(x))
        # x = F.dropout(x, p=0.5) 
        # x = self.L2p(x)

        # # Add the tensors
        # pt_addition_result = torch.add(x1p, x2p)
        # # Compute the mean
        # weighted_mean = torch.mean(pt_addition_result)
        # x  = weighted_mean + x


        if self.include_top:
            cls_ = self.classifier((x))
        return x, cls_
