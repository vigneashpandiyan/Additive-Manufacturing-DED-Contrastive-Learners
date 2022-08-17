
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:48:47 2022

@author: srpv
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools
import os
from sklearn import metrics
import pydot
import collections
import pydotplus
import os
import pydotplus
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import joblib
from sklearn.model_selection import cross_val_score
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split# implementing train-test-split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Plots import *

#%%

def LR(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=1000, random_state=123)
    model.fit(X_train,y_train)
    
    
    
    
    predictions = model.predict(X_test)
    print("LogisticRegression Accuracy:",metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    
    
    
    graph_name1= 'LR'+'_without normalization w/o Opt'
    graph_name2=  'Logistic Regression'
    
    graph_1= 'LR'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2= 'LR'+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    
    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]
    
    for title, normalize ,graphname  in titles_options:
        plt.figure(figsize = (20, 10),dpi=400)
        disp = plot_confusion_matrix(model, X_test, y_test,
                                      display_labels=['P1', 'P2', 'P3', 'P4','P5','P6'],
                                      cmap=plt.cm.RdPu,xticks_rotation='vertical',
                                    normalize=normalize)
        plt.title(title, size = 12)
        
        plt.savefig(graphname,bbox_inches='tight',dpi=400)
    savemodel=  'LR'+'_model'+'.sav'    
    joblib.dump(model, savemodel)
    
#%%

train_embeddings = 'train_embeddings'+'_'+ '.npy'
train_labelsname = 'train_labels'+'_'+'.npy'
test_embeddings = 'test_embeddings'+'_'+ '.npy'
test_labelsname = 'test_labels'+'_'+'.npy'


X_train = np.load(train_embeddings).astype(np.float64)
y_train = np.load(train_labelsname).astype(np.float64)
y_train=Dataframe_Manipulation_Classifier(y_train)

X_test = np.load(test_embeddings).astype(np.float64)
y_test = np.load(test_labelsname).astype(np.float64)
y_test=Dataframe_Manipulation_Classifier(y_test)

#%%

LR(X_train, X_test, y_train, y_test)
