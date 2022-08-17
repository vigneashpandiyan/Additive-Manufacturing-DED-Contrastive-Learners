# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:48:47 2022

@author: srpv
"""


from torchvision.datasets import MNIST
from torchvision import transforms

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
from os import walk
import os

import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 
import torchvision 
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from datasets import SiameseMNIST
from datasets import SiamesePlot

from matplotlib import animation
from Plots import *

#%%
# setting the root directories and categories of the images
datadir = 'C:/Users/srpv/Desktop/C4 Science/DED Data/'
traindir = datadir + 'train/'
testdir = datadir + 'test/'

categories = [[folder, os.listdir(traindir + folder)] for folder in os.listdir(traindir)  if not folder.startswith('.') ]

mnist_classes = ['1', '2', '3', '4','5','6']

#%%

# Datasize preparation
# Set up data loaders

transformations = transforms.Compose([
        torchvision.transforms.Resize((480,320)),
        transforms.ToTensor()])

    
train_loader_folder = datasets.ImageFolder(root=traindir, transform=transformations)
test_loader_folder = datasets.ImageFolder(root=testdir, transform=transformations)

#%%

batch_size = 100
kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
trainloader = torch.utils.data.DataLoader(train_loader_folder, batch_size=batch_size, shuffle=True, **kwargs)
testloader = torch.utils.data.DataLoader(test_loader_folder, batch_size=batch_size, shuffle=True, **kwargs)


dataSize = len(trainloader.dataset) # self-defined dataset size
TRAIN_PCT = 0.9 # percentage of entire dataset for training
train_size = int(dataSize * TRAIN_PCT)
val_size = dataSize - train_size


siamese_train_dataset = SiameseMNIST(categories, traindir, dataSize, transformations) # Returns pairs of images and target same/different
train_set, val_set = random_split(siamese_train_dataset, [train_size, val_size])


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)

#%%

# Setting up the network and training parameters

from networks import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss

margin = 1.
embedding_net = EmbeddingNet(dropout=0.1)
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

optimizer =  torch.optim.SGD(model.parameters(),lr=0.0005,momentum=0.9)
scheduler = StepLR(optimizer, step_size = 200, gamma= 0.25 )

n_epochs = 1
log_interval = 25

#%%
#Training of the Network

train_losses,val_losses=fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


train_plot = 'train_losses'+'_'+ '.npy'
val_plot = 'val_losses'+'_'+'.npy'


np.save(train_plot,train_losses, allow_pickle=True)
np.save(val_plot,val_losses, allow_pickle=True)


#%%



siamese_train = SiamesePlot(categories, traindir, dataSize, transformations) # Returns pairs of images and target same/different

batch_size = 256
kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
train_loader_1 = torch.utils.data.DataLoader(siamese_train, batch_size=batch_size, shuffle=True, **kwargs)


for img1, img2, label in train_loader_1:
    print()
    if label[0] == 1.0:
        
        plt.subplot(1,2,1)
        plt.imshow(np.transpose(img1[0][0]))
        
        plt.subplot(1,2,2)
        plt.imshow(np.transpose(img2[0][0]))
       
        plt.savefig('Same.png', dpi=600,bbox_inches='tight')
        plt.show()
       
        break
    

for img1, img2, label in train_loader_1:
    print()
    if label[0] == 0.0:
        # print(img1[0])
        plt.subplot(1,2,1)
        plt.imshow(np.transpose(img1[0][0]))
        
        plt.subplot(1,2,2)
        plt.imshow(np.transpose(img2[0][0]))
        
        plt.savefig('Different.png', dpi=600,bbox_inches='tight')
        plt.show()
        break

#%%
        

plt.xlabel('Epoch')
plt.ylabel('Loss values')
plt.plot(train_losses, c='b', label='Training Loss',linewidth =2.0)
# plt.plot(val_losses, label="Validation Loss",linewidth =2.0)
plt.legend( loc='upper right')
plt.savefig('Siamese_Loss_Siamese.png', dpi=600,bbox_inches='tight')
plt.show()

#%%

color = [ 'purple','orange','green','red', 'blue', 'cyan']
marker= ["d","s","*",">","X","o"]

mnist_classes = ['P1', 'P2', 'P3', 'P4','P5','P6']
graph_name_2D='Training_Feature_2D' +'_'+'.png'
graph_title = "Feature space distribution"

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(7,5))
    j=0
    for i in range(len(train_loader_folder.classes)):
        print(i)
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.7, color=color[j],marker=marker[j],s=100)
        j=j+1
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes,bbox_to_anchor=(1.32, 1.05))
    plt.xlabel ('Weights_1', labelpad=10)
    plt.ylabel ('Weights_2', labelpad=10)
    plt.title(str(graph_title),fontsize = 15)
    plt.savefig(graph_name_2D, bbox_inches='tight',dpi=600)
    plt.show()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 4))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images1 = images
            target=target
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

PATH = './Contrastive.pth'
torch.save(model.state_dict(), PATH)
torch.save(model, PATH)




#%%

train_embeddings_baseline, train_labels_baseline = extract_embeddings(trainloader, model)

train_embeddings_baseline=train_embeddings_baseline.astype(np.float64)
train_labels_baseline=train_labels_baseline.astype(np.float64)
train_embeddings = 'train_embeddings'+'_'+ '.npy'
train_labelsname = 'train_labels'+'_'+'.npy'
np.save(train_embeddings,train_embeddings_baseline, allow_pickle=True)
np.save(train_labelsname,train_labels_baseline, allow_pickle=True)
plot_embeddings(train_embeddings_baseline, train_labels_baseline)

test_embeddings_baseline, test_labels_baseline = extract_embeddings(testloader, model)

test_embeddings_baseline=test_embeddings_baseline.astype(np.float64)
test_labels_baseline=test_labels_baseline.astype(np.float64)
test_embeddings = 'test_embeddings'+'_'+ '.npy'
test_labelsname = 'test_labels'+'_'+'.npy'
np.save(test_embeddings,test_embeddings_baseline, allow_pickle=True)
np.save(test_labelsname,test_labels_baseline, allow_pickle=True)



#%%

    
graph_name='Training_Feature' +'_'+'.png'
ax,fig=Three_embeddings(train_embeddings_baseline, train_labels_baseline,graph_name,ang=45)
gif1_name= str('Training_Feature')+'.gif'

#%%
def rotate(angle):
      ax.view_init(azim=angle)
      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(gif1_name, writer=animation.PillowWriter(fps=20))

