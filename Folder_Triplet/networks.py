# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:48:47 2022

@author: srpv
"""


import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):    
    def __init__(self,dropout):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3),
                                     nn.BatchNorm2d(4),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3),
                                     nn.BatchNorm2d(8),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout))

        self.fc = nn.Sequential(nn.Linear(64 *8 * 13, 512),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(512, 64),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(64, 4)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(-1, 64 *8 * 13)
        output = self.fc(output)
        return output
    
    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
