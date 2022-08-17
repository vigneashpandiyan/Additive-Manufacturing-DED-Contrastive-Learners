# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:48:47 2022

@author: srpv
"""



import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import random
from os import walk
import torch

# Data--> https://polybox.ethz.ch/index.php/s/HUcJ7cJ18K0MrEn 
# root_dir = '../Data/train/'   #place in a folder

root_dir = 'C:/Users/srpv/Desktop/C4 Science/DED Data/train/'

categories = [[folder, os.listdir(root_dir + folder)] for folder in os.listdir(root_dir)  if not folder.startswith('.') ]


# creating the pairs of images for inputs, same character label = 1, vice versa
class TripletMNIST(Dataset):
    def __init__(self, categories, root_dir, setSize, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.transform = transform
        self.setSize = setSize
    def __len__(self):
        return self.setSize
    def __getitem__(self, idx):
        img1 = None
        img2 = None
        img3 = None
        
        category = random.choice(categories)
        character = random.choice(category[0])
        imgDir = root_dir + character
        img1Name = random.choice(os.listdir(imgDir))
        img2Name = random.choice(os.listdir(imgDir))
        img1 = Image.open(imgDir + '/' + img1Name)
        img2 = Image.open(imgDir + '/' + img2Name)
        
        category1 = random.choice(categories)
        
        
        if category1 == category:
                 while True:
                     category1 = random.choice(categories)
                     if category1 not in category1:
                        category1 = random.choice(categories)
                        
                        break
        
        category1 = random.choice(category1[0])
        imgDir2 = root_dir + category1
        img3Name = random.choice(os.listdir(imgDir2))
            
        img3 = Image.open(imgDir2 + '/' + img3Name)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3) , []         


# creating the pairs of images for inputs, same character label = 1, vice versa
class TripletPlot(Dataset):
    def __init__(self, categories, root_dir, setSize, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.transform = transform
        self.setSize = setSize
    def __len__(self):
        return self.setSize
    def __getitem__(self, idx):
        img1 = None
        img2 = None
        img3 = None
        
        category = random.choice(categories)
        character = random.choice(category[0])
        imgDir = root_dir + character
        img1Name = random.choice(os.listdir(imgDir))
        img2Name = random.choice(os.listdir(imgDir))
        img1 = Image.open(imgDir + '/' + img1Name)
        img2 = Image.open(imgDir + '/' + img2Name)
        
        category1 = random.choice(categories)
        category1 = random.choice(categories)
        character1= random.choice(category1[0])
        imgDir1 = root_dir + character1
        img3Name = random.choice(os.listdir(imgDir1))
        
        while imgDir == imgDir1:
            category1 = random.choice(categories)
            character1= random.choice(category1[0])
            imgDir1 = root_dir + character1
            img3Name = random.choice(os.listdir(imgDir1))
            
        img3 = Image.open(imgDir1 + '/' + img3Name)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img1, img2, img3        



