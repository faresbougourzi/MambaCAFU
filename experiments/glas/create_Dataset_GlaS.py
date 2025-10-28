# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 06:21:39 2023

@author: VinhTH
"""

import torch
import cv2
import os
from sklearn.model_selection import KFold
random_state= 44 # 43 # 42
Kf=KFold(n_splits=5,shuffle=True)
from sklearn.model_selection import train_test_split


import csv
database_path = 'data_sets/GlaS'

Train_csv = f'{database_path}/Grade.csv'  

with open(Train_csv , newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
img_link = []
mask_link = []

print(f"**** A total of images {len(data)} ****")
    
for i in range(len(data)-1):
    img_link.append(data[i+1][0]+'.bmp')
    mask_link.append(data[i+1][0]+'_anno.bmp')

val_pathsave = f"{database_path}/val" 
if not os.path.exists(val_pathsave):
    os.makedirs(val_pathsave) 




indicies = list(range(len(img_link)))
for fold,(train_idx,test_idx)in enumerate(Kf.split(indicies)):
    fold=fold+1
    print(fold)
 

################
    Training_img = []
    Validation_img = []

    
    Training_maskl = []
    Validation_maskl = []
    
    ind_tr=-1
    ind_val=-1
    ind_test=-1

    im=-1          
    for ind in range(len(img_link)):
        im+=1
        if im in train_idx:
            ind_tr+= 1
            im_path = os.path.join(os.path.join(database_path,img_link[ind]))
            img1 = cv2.imread(im_path)
            
            im_path3 = os.path.join(database_path,mask_link[ind])
            lesion_mask = cv2.imread(im_path3, cv2.IMREAD_GRAYSCALE) 
    
            img1=cv2.resize(img1, (512,512), interpolation = cv2.INTER_AREA)
            lesion_mask=cv2.resize(lesion_mask, (512,512), interpolation = cv2.INTER_AREA)
    
            Training_img.append(img1)
            Training_maskl.append(lesion_mask)             
            # torch.save(pt_data, name)

        if im in test_idx:
            ind_test+= 1
            im_path = os.path.join(os.path.join(database_path,img_link[ind]))
            img1 = cv2.imread(im_path)
            
            im_path3 = os.path.join(database_path,mask_link[ind])
            lesion_mask = cv2.imread(im_path3, cv2.IMREAD_GRAYSCALE) 
    
            img1=cv2.resize(img1, (512,512), interpolation = cv2.INTER_AREA)
            lesion_mask=cv2.resize(lesion_mask, (512,512), interpolation = cv2.INTER_AREA)
    
            img1=cv2.resize(img1, (512,512), interpolation = cv2.INTER_AREA)
            lesion_mask=cv2.resize(lesion_mask, (512,512), interpolation = cv2.INTER_AREA)
    
            Validation_img.append(img1)
            Validation_maskl.append(lesion_mask) 
            
    import torch
    X = [i for i in Training_img]
    y = [i for i in Training_maskl] 

    training= (X, y)
    torch.save(training,f'{database_path}/Train_Fold'+str(fold)+'.pt') 
    
    
    import torch
    X = [i for i in Validation_img]
    y = [i for i in Validation_maskl] 

    training= (X, y)
    torch.save(training,f'{database_path}/Test_Fold'+str(fold)+'.pt')             

if __name__ == "__main__" :
    filename = f"{database_path}/Train_Fold1.pt"
    import numpy as np
    train_set = torch.load(filename)
    X, y = train_set
    y1 = y[0]
    X = X[0]
    print(np.unique(y1))
    print(X.shape)