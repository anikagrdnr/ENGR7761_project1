"""
Class structure to encapsulate data methods 
TODO fix pathways to data, labels 

"""

from torch.utils.data import DataLoader
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch
from collections import Counter
import matplotlib.pyplot as plt



def select_transforms(img_size, train=True, light=false):

    rotation=

    light=

    #etc to easily select







def get_transforms(img_size):
    """
    uses pytorch Compose to get transforms
    defines train and test transforms
    Key idea - add real world distortions to dataset to enable detection under different conditions (rotation, motion, lighting)
    Given object detection domain such that shape, colour, texture all relevant to classification
    TODO: mod for ease of transform selection for testing 
    """
    train_transforms=transforms.Compose([

        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  

        #rotation invariance
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        #lighting
        transforms.RandomApply([
        transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1 
            )
        ], p=0.7),
        #motion
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
        # texture/shape regularisation — keep probability low, colour matters
        transforms.RandomGrayscale(p=0.1)
    ])

        # already normalise — ImageNet stats 
        #transforms.Normalize(
        # mean=[0.485, 0.456, 0.406],
            #std=[0.229, 0.224, 0.225]
        #    ),
        # texture boundaries — Andrearczyk & Whelan (2016)
        #LaplacianChannel(),                             # (3, H, W) -> (4, H, W)
        # material properties — periodic texture
        #FourierChannel(),  

# test — no distortions, only what is strictly necessary
    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        #transforms.Normalize( #already normalised
            #mean=[0.485, 0.456, 0.406],
            #std=[0.229, 0.224, 0.225]
        #LaplacianChannel(),
        #FourierChannel(),    ]),
    ])
    return train_transforms, test_transforms


class WasteData(Dataset):
    """
    class to encapsulate all database methods 
    TODO: add dataset file imports 

    """

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        TODO: pytorch implementation to mod
        instantiates dataset object
        """
        self.img_labels = pd.read_csv(annotations_file) #replace with kaggle load in
        self.img_dir = img_dir
        self.transform = transform #add transforms using compose (note: laplacian, fourier not in there needing different tensor dimensionality)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels) #double check best way to access length given dataset

    def __getitem__(self, idx):
        """
        TODO: adjust once file location determined
        -> add labels?
        -> add transform
        """    
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def datasetAnalysis():
        """
        get bias and iid of data 
        """
        #uses imageFolder count of targets 
        counts = Counter(self.data.targets)

        classes = self.data.classes #ADD LABELS ETC.. fix access one dataset path revealed
        
        plt.bar(classes, [counts[i] for i in range(len(classes))])
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Dataset Class Distribution')
        plt.tight_layout()
        plt.show()


#instead of get_batch
def get_dataloader(dataset, labels, batch_size, shuffle=False):
    """
    Uses weighted sample to select mini-batch
    -> ensuring iid by weighting to remove bias (select from uniform distribution) 
    -> randomly samples 
    -> uses dataloader to apply randomised minibatch selection 
    -> calculates loss per mini-batch
    MINI-BATCH affect on loss landscape:
        - larger mini batch, smaller step

    """
    counts=torch.bincount(torch.tensor(labels))
    weights=1.0/counts[labels] #weights according to magnitude of ea class
    sampler=WeightedRandomSampler(weights, num_samples=len(weights))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


