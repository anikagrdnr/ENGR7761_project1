"""
Model Builder Class
-> copied from notebook starting template
"""

from tqdm.notebook import trange, tqdm
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np
import zipfile
import imageio
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from torch.distributions import Normal
from torch.autograd import Variable


#CNN class 
class CNN(nn.Module):
    
    #init 
    def __init__(self, hidden_size, nb_class):
        """
        define architecture of CNN
            - 3 layers
            - activation function
            - pooling
        """
                
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        self.linear1 = nn.Linear( 2304, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, nb_class)

        self.cnn1.weight.data.normal_(mean=0, std=3e-4)
        self.cnn1.bias.data.fill_(0.1)

        self.cnn2.weight.data.normal_(mean=0, std=3e-4)
        self.cnn2.bias.data.fill_(0.1)

        self.cnn3.weight.data.normal_(mean=0, std=3e-4)
        self.cnn3.bias.data.fill_(0.1)

        self.linear1.weight.data.normal_(mean=0, std=3e-4)
        self.linear1.bias.data.fill_(0.1)

        self.linear2.weight.data.normal_(mean=0, std=3e-4)
        self.linear2.bias.data.fill_(0.1)

        self.linear_out.weight.data.normal_(mean=0, std=3e-4)
        self.linear_out.bias.data.fill_(0.1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.bn4 = nn.LayerNorm(hidden_size)
        self.bn5 = nn.LayerNorm(hidden_size)

    def forward(self, images):

        images = images.reshape([3, 256, 256])
        images = np.float32(images)
        images = torch.FloatTensor(images).to(device).unsqueeze(0)

        y_pred = f.relu(self.cnn1(images))
        y_pred = self.bn1(y_pred)
        y_pred = self.pool(y_pred)

        y_pred = f.relu(self.cnn2(y_pred))
        y_pred = self.bn2(y_pred)
        y_pred = self.pool(y_pred)

        y_pred = f.relu(self.cnn3(y_pred))
        y_pred = self.bn3(y_pred)
        y_pred = self.pool(y_pred)

        y_pred = y_pred.reshape([1, 2304])

        y_pred = f.relu(self.linear1(y_pred))
        y_pred = self.bn4(y_pred)
        y_pred = f.relu(self.linear2(y_pred))
        y_pred = self.bn5(y_pred)

        y_pred = self.linear_out(y_pred)

        return y_pred

"""
other methods to analyse complexity etc. 
"""
#other methods 
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(y_pred, y):

    _, index_y = torch.max(y, dim=-1)
    _, index_y_pred = torch.max(y_pred[0], dim=-1)

    if index_y == index_y_pred:

        acc = 1

    else:

        acc = 0

    return acc

def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
    

