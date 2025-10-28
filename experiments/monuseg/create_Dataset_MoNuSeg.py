# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 04:32:24 2023

@author: VinhTH
"""


import torch
import cv2
import os
from sklearn.model_selection import KFold
Kf=KFold(n_splits=5,shuffle=True)
from sklearn.model_selection import train_test_split
import re
random_state= 14 #13 #12
root_path = 'data_sets/MoNuSeg'
database_path1 = f'{root_path}/Test_Folder'
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

img_link = []
mask_link = []

database_path = f'{root_path}/Test_Folder/img'
data_splits = sorted_alphanumeric(os.listdir(os.path.join(database_path)))
   
for im_name in data_splits:
    img_link.append(f'{root_path}/Test_Folder/img/'+ im_name)
    mask_link.append(f'{root_path}/Test_Folder/labelcol/'+ im_name[:-3]+'png')
    
database_path = f'{root_path}/Train_Folder/img'
data_splits = sorted_alphanumeric(os.listdir(os.path.join(database_path)))
   
for im_name in data_splits:
    img_link.append(f'{root_path}/Train_Folder/img/'+ im_name)
    mask_link.append(f'{root_path}/Train_Folder/labelcol/'+ im_name[:-3]+'png')  
    
database_path = f'{root_path}/Val_Folder/img'
data_splits = sorted_alphanumeric(os.listdir(os.path.join(database_path)))
   
for im_name in data_splits:
    img_link.append(f'{root_path}/Val_Folder/img/'+ im_name)
    mask_link.append(f'{root_path}/Val_Folder/labelcol/'+ im_name[:-3]+'png')     
    

val_pathsave = f'{root_path}/DatasMoNuSeg/'
if not os.path.exists(val_pathsave):
    os.makedirs(val_pathsave) 



indicies = list(range(len(img_link)))
print("total = ", len(img_link))
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
            im_path = img_link[ind]
            img1 = cv2.imread(im_path)
            
            im_path3 = mask_link[ind]
            lesion_mask = cv2.imread(im_path3, cv2.IMREAD_GRAYSCALE) 
    
            img1=cv2.resize(img1, (512,512), interpolation = cv2.INTER_AREA)
            lesion_mask=cv2.resize(lesion_mask, (512,512), interpolation = cv2.INTER_AREA)
    
            Training_img.append(img1)
            Training_maskl.append(lesion_mask)             

        if im in test_idx:
            ind_test+= 1
            im_path = img_link[ind]
            img1 = cv2.imread(im_path)
            
            im_path3 = mask_link[ind]
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
    torch.save(training,f'{val_pathsave}/Train_Fold'+str(fold)+'.pt') 
    
    
    import torch
    X = [i for i in Validation_img]
    y = [i for i in Validation_maskl] 

    training= (X, y)
    torch.save(training,f'{val_pathsave}/Test_Fold'+str(fold)+'.pt')  