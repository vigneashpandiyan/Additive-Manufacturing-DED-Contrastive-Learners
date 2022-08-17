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

# setting the root directories and categories of the images
# Data--> https://polybox.ethz.ch/index.php/s/HUcJ7cJ18K0MrEn 
# root_dir = '../Data/train/'   #place in a folder

root_dir = 'C:/Users/srpv/Desktop/C4 Science/DED Data/train/'

categories = [[folder, os.listdir(root_dir + folder)] for folder in os.listdir(root_dir)  if not folder.startswith('.') ]

   

# creating the pairs of images for inputs, same character label = 1, vice versa
class SiameseMNIST(Dataset):
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
        label = None
        should_get_same_class = random.randint(0,1)
        if should_get_same_class: # select the same character for both images
            category = random.choice(categories)
            character = random.choice(category[0])
            imgDir = root_dir + character
            img1Name = random.choice(os.listdir(imgDir))
            img2Name = random.choice(os.listdir(imgDir))
            img1 = Image.open(imgDir + '/' + img1Name)
            img2 = Image.open(imgDir + '/' + img2Name)
            # print(imgDir+'/'+img1Name)
            # print(imgDir+'/'+img2Name)
            label = 1
        else: # select a different character for both images
            category1 = random.choice(categories)
            
            while True:
                category2 = random.choice(categories)
                if category2 not in category1:
                    category2 = random.choice(categories)
                    break
            
            #category1, category2 = random.choice(categories), random.choice(categories)
            character1, character2 = random.choice(category1[0]), random.choice(category2[0])
            imgDir1, imgDir2 = root_dir + character1, root_dir + character2
            img1Name = random.choice(os.listdir(imgDir1))
            img2Name = random.choice(os.listdir(imgDir2))
            if img1Name == img2Name:
                 while True:
                     category2 = random.choice(categories)
                     if category2 not in category1:
                        category2 = random.choice(categories)
                        character2 = random.choice(category2[0])
                        imgDir2 = root_dir + character2
                        img2Name = random.choice(os.listdir(imgDir2))
                        break
            
            label = 0
            img1 = Image.open(imgDir1 + '/' + img1Name)
            img2 = Image.open(imgDir2 + '/' + img2Name)
#         plt.imshow(img1)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # return (img1, img2), torch.from_numpy(np.array([label], dtype=np.float32))
        return (img1, img2), should_get_same_class         
    
class SiamesePlot(Dataset):
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
        label = None
        if idx % 2 == 0: # select the same character for both images
            category = random.choice(categories)
            character = random.choice(category[0])
            imgDir = root_dir + character
            img1Name = random.choice(os.listdir(imgDir))
            img2Name = random.choice(os.listdir(imgDir))
            img1 = Image.open(imgDir + '/' + img1Name)
            img2 = Image.open(imgDir + '/' + img2Name)
            # print(imgDir+'/'+img1Name)
            # print(imgDir+'/'+img2Name)
            label = 1.0
        else: # select a different character for both images
            category1, category2 = random.choice(categories), random.choice(categories)
            category1, category2 = random.choice(categories), random.choice(categories)
            character1, character2 = random.choice(category1[0]), random.choice(category2[0])
            imgDir1, imgDir2 = root_dir + character1, root_dir + character2
            img1Name = random.choice(os.listdir(imgDir1))
            img2Name = random.choice(os.listdir(imgDir2))
            while img1Name == img2Name:
                img2Name = random.choice(os.listdir(imgDir2))
            label = 0.0
            img1 = Image.open(imgDir1 + '/' + img1Name)
            img2 = Image.open(imgDir2 + '/' + img2Name)
#         plt.imshow(img1)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))    


