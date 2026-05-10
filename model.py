"""
MLCV Project 1 — model.py
CNN architecture for 10-class waste classification.
-> adjusted from OG architecture:
    - He initialisation 
    - nested function call implementation 
    - 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    3-layer CNN for waste image classification.

    Architecture:
        Conv → BN → ReLU → MaxPool  (x3)
        AdaptiveAvgPool
        Flatten
        Linear → BN → ReLU (x2)
        Linear → logits

    Args:
        hidden_size : number of neurons in fully-connected layers
                      (tune via config.py — try 256, 512, 1024)
        nb_class    : number of output classes (10)
    """

    def __init__(self, hidden_size, nb_class):
        super(CNN, self).__init__()

        #reduced stride to 1, added padding 
        self.conv1 = nn.Conv2d(3,   64,  kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        #downsampling
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # AdaptiveAvgPool forces output to 3×3 regardless of input size
        # -> removes hard-coded flatten dimension
        # -> flatten size is always 256 * 3 * 3 = 2304
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

        # Applied after conv, before ReLU (Conv → BN → ReLU)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.ln4 = nn.LayerNorm(hidden_size) #layernorm 
        self.ln5 = nn.LayerNorm(hidden_size) #layernorm 

        #adjust initial parameter of linear layer 
        self.linear1    = nn.Linear(256 * 3 * 3, hidden_size)
        self.linear2    = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, nb_class)

        # step 2 regularisation
        # self.dropout = nn.Dropout(p=0.5)
        #x=self.dropout # assigns dropout to x 

        # Kaiming (He) initialisation for ReLU networks
        # Replaces original std=3e-4 which caused vanishing gradients
        self._init_weights()

    #uses weight initialisation He for relu only 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """
        Args:
            x : tensor of shape [B, 3, H, W] — batch from DataLoader
        Returns:
            logits : tensor of shape [B, nb_class]
        """
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Adaptive pool + flatten
        x = self.adaptive_pool(x)
        x = x.flatten(1)

        # FC block 1
        x = F.relu(self.ln4(self.linear1(x)))
        # self.dropout(x)   # uncomment for step 2

        # FC block 2
        x = F.relu(self.ln5(self.linear2(x)))
        # self.dropout(x)   # uncomment for step 2

        return self.linear_out(x)