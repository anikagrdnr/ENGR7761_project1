https://www.sciencedirect.com/science/article/pii/S2773186324000768"""
Project 1: MLCV

CNN for rubbish classification 
reference "a study of garbage classification with CNN" for first test archietcture 

? wat transforms 
- grayscale
-fourier
-etc

? initialisation
- weights (guassian, xavier, he, lecunn)
- bias 

? activation functions 

init Includes
- 3 x conv layers 
- relu activation
- pooling 
- adam gradient 
< 2hrs to solve 

TODO
-adjust initial set up 
-change val method 
-report (only results is graded)

-> new function to make iid 

**upload to github**

"""
from modelBuilder import *
from dataloader import *


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

import os
from pathlib import Path
import random

if __name__=="__main__":    
    # Use Cuda GPU, if not available CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: " + str(device))

    #import image and dataset 
    img = Image.open('standardized_256/plastic/plastic_45.jpg')
    plt.imshow(img)

    #set up nn 

    #train and print/ output graph of losses 



