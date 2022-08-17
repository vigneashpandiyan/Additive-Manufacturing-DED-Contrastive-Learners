# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:48:47 2022

@author: srpv
"""


import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns


plt.rc("font", size=15)

def Three_embeddings(embeddings, targets,graph_name,ang, xlim=None, ylim=None):
    group=targets
    
    df2 = pd.DataFrame(group) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
    df2=df2['Categorical'].replace(4,'P5')
    df2 = pd.DataFrame(df2) 
    df2=df2['Categorical'].replace(5,'P6')
    group = pd.DataFrame(df2) 
    
    group=group.to_numpy()
    group = np.ravel(group)
    
    
    x1=embeddings[:, 0]
    x2=embeddings[:, 1]
    x3=embeddings[:, 2]
    
    
    df = pd.DataFrame(dict(x=x1, y=x2,z=x3, label=group))
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    #uniq=["0","1","2","3"]
    
    
    fig = plt.figure(figsize=(12,6), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2
   
    ax = plt.axes(projection='3d')
    
    ax.grid(False)
    ax.view_init(azim=ang)#115
    
    color = [ 'purple','orange','green','red', 'blue', 'cyan']
    marker= ["d","s","*",">","X","o"]
    
    
    ax.set_facecolor('white') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    graph_title = "Feature space distribution"
    
    j=0
    for i in uniq:
        print(i)
        indx = group == i
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
        ax.plot(a, b, c ,color=color[j],label=uniq[j],marker=marker[j],linestyle='',ms=7)
        j=j+1
     
    plt.xlabel ('Dimension-1', labelpad=10)
    plt.ylabel ('Dimension-2', labelpad=10)
    ax.set_zlabel('Dimension-3',labelpad=10)
    plt.title(str(graph_title),fontsize = 15)
    
    plt.legend(markerscale=20)
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    #plt.zticks(fontsize = 25)
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(graph_name, bbox_inches='tight',dpi=400)
    plt.show()
    return ax,fig

def Dataframe_Manipulation(Distance,target):
        
       
    df1 = pd.DataFrame(Distance) 
    df1.columns = ['Distance']
    df2 = pd.DataFrame(target) 
    df2.columns = ['Categorical']
    
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
    df2=df2['Categorical'].replace(4,'P5')
    df2 = pd.DataFrame(df2) 
    df2=df2['Categorical'].replace(5,'P6')
    df2 = pd.DataFrame(df2)  
    
    df=pd.concat([df1,df2], axis=1)
    new_columns = list(df.columns)
    new_columns[-1] = 'Target'
    df.columns = new_columns
    df.Target.value_counts()
    df = df.sample(frac=1.0)
    
    print(df.shape)
    
    return df


def Dataframe_Manipulation_Classifier(target):
        
       
    
    df2 = pd.DataFrame(target) 
    df2.columns = ['Categorical']
    
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
    df2=df2['Categorical'].replace(4,'P5')
    df2 = pd.DataFrame(df2) 
    df2=df2['Categorical'].replace(5,'P6')
    df2 = pd.DataFrame(df2)  
    
    
    
    return df2

