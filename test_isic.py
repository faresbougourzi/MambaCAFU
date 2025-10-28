# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:39:50 2023

"""
import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as TF
import os, random

from sklearn.metrics import roc_auc_score, jaccard_score
from experiments.isic.ISIC_data_loader import *

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
import os.path

import os
import ml_collections
import cv2
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,default='MambaCAFU_v2', choices=['MambaCAFU_v1','MambaCAFU_v2']) 

parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--runs', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--itrsave', type=int,
                    default=0, help='output channel of network')
parser.add_argument('--base_lr', type=float,  default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--batch_size', type=int,
                    default=18, help='training Batch Size')

parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used') 

args = parser.parse_args()

args.exp = args.model_name
if args.exp == 'MambaCAFU_v1':
    import  networks.network as MyPYNet
    model  = MyPYNet.MambaCAFU(in_channels=3, num_classes= args.num_classes, version = 'v0').cuda()
    snapshot_path = "./model/ISIC17/"+args.exp
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
elif args.exp == 'MambaCAFU_v2':
    import  networks.network as MyPYNet
    model  = MyPYNet.MambaCAFU(in_channels=3, num_classes= args.num_classes, version = 'v2').cuda()
    snapshot_path = "./model/ISIC17/"+args.exp
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)  


args.ModelNameSave = args.model_name
dataset_idx = 'ISIC2017'

database_path = 'data_sets/ISIC2017/processed/'

print("**** Config ****", args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


###### Losses #################################
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

from torch.nn.modules.loss import CrossEntropyLoss

########################################################
#######################################################

    
if __name__ =="__main__":
    F1_mean, dise_mean, IoU_mean = [], [], []
    root_path = "./model/"
    for itr in range(args.runs):
        # itr += args.itrsave
        # model_sp = root_path +  args.ModelNameSave + "/Models"
        # if not os.path.exists(model_sp):
        #     os.makedirs(model_sp)
        ############################
        
        # name_model_final = model_sp+ '/' + str(itr) + '_fi.pt'
        # name_model_bestF1 =  model_sp+ '/' + str(itr) + '_bt.pt'

        token = "bestdsc"
        mode = 'bt' # "fi" "valloss"
        
        model_spR = root_path  + args.ModelNameSave + "ISIC2017"
        if not os.path.exists(model_spR):
            os.makedirs(model_spR)
            
        training_tsx = model_spR+ f'/test_result_0_{mode}_{token}.txt'  
        
        snapshot_weight = f'outputs/model/ISIC2017/{args.ModelNameSave}/Models/0_{mode}_{token}.pt' # #0_valloss_bs12lr001.pt, 0_bt_bs12lr001.pt
        print("[DEBUG][WEIGHT] Load model's weight = ", snapshot_weight)
        model.load_state_dict(torch.load(snapshot_weight))
        
        
        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(args.num_classes)
        
        validate_set   = isic_loader(path_Data = database_path, train = False, Test=True)
        print("[DEBUG] The number of samples ", len(validate_set))

        ###############################################

        
        torch.set_grad_enabled(True)    
        ############################
        # Part 5
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=30, shuffle=False)
        
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        
        start = time.time()
        model.to(device)
        
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        train_dise, valid_dise = [], []
        train_dise2, valid_dise2 = [], []
        
        train_IoU, valid_IoU = [], []
        
        train_F1score, valid_F1score = [], []
        
        train_Spec, valid_Spec = [], []
        train_Sens, valid_Sens = [], []
        train_Prec, valid_Prec = [], []
            
        epoch_count = []
        
        best_Dice = -1
        epochs = args.max_epochs
        iter_num = -1
        iters = len(validate_loader)
        LR = args.base_lr
        max_iterations = len(validate_loader)* args.max_epochs
    
        for epoch in range(args.max_epochs):
            epoch_count.append(epoch)

            for phase in ['test']:
                if phase == 'test':
                    model.train(False)  # Set model to evaluate mode
                    dataloader = validate_loader
        
                running_loss = 0.0
        
                num_correct = 0
                num_pixels = 0
        
                step = 0
        
                # iterate over data
                dice_scores = 0
                dice_scores2 = 0
                iou_pred = 0
                TP = 0
                TN = 0
                FP = 0
                FN = 0
                iij = -1
                for batch in tqdm(dataloader):
                    x, y1 = batch['image'], batch['mask']
                    x = x.to(device)
                    y1 = y1.long().to(device)    
                    
        
                    step += 1
        
                    # forward pass
                    if phase == 'test':

                        with torch.no_grad():
                            outputs11 = model(x)
                            # calculate the loss 
                            loss = ce_loss(outputs11, y1.long())   
        
                    running_loss += loss
                    
                    preds = torch.argmax(outputs11, 1) 
                    
                    preds = preds.squeeze(dim=1).cpu().numpy().astype(int)
                    
                    yy = y1 > 0.5
                    yy = yy.squeeze(dim=1).cpu().numpy().astype(int)
        
                    num_correct += np.sum(preds == yy)
        
                    TP += np.sum(((preds == 1).astype(int) +
                                (yy == 1).astype(int)) == 2)
                    TN += np.sum(((preds == 0).astype(int) +
                                (yy == 0).astype(int)) == 2)
                    FP += np.sum(((preds == 1).astype(int) +
                                (yy == 0).astype(int)) == 2)
                    FN += np.sum(((preds == 0).astype(int) +
                                (yy == 1).astype(int)) == 2)
                    num_pixels += preds.size
                    for idice in range(preds.shape[0]):
                        dice_scores += (2 * (preds[idice] * yy[idice]).sum()) / (
                            (preds[idice] + yy[idice]).sum() + 1e-8
                        )
        
                    predss = np.logical_not(preds).astype(int)
                    yyy = np.logical_not(yy).astype(int)
                    for idice in range(preds.shape[0]):
                        dice_sc1 = (2 * (preds[idice] * yy[idice]).sum()) / (
                            (preds[idice] + yy[idice]).sum() + 1e-8
                        )
                        dice_sc2 = (2 * (predss[idice] * yyy[idice]).sum()) / (
                            (predss[idice] + yyy[idice]).sum() + 1e-8
                        )
                        dice_scores2 += (dice_sc1 + dice_sc2) / 2
                        
                    for idice in range(preds.shape[0]):    
                        iou_pred += jaccard_score(preds[idice].reshape(-1), yy[idice].reshape(-1))
        
                    del x
                    del y1
        
                epoch_loss = running_loss / len(dataloader.dataset)
        
                epoch_acc2 = (num_correct/num_pixels)*100
                epoch_dise = dice_scores/len(dataloader.dataset)
                epoch_dise2 = dice_scores2/len(dataloader.dataset)
        
                Spec = 1 - (FP/(FP+TN))
                Sens = TP/(TP+FN)  # Recall
                Prec = TP/(TP+FP + 1e-8)
                # F1score = 2 *(Sens*Prec) / (Sens+Prec+ 1e-8)
                F1score = TP / (TP + ((1/2)*(FP+FN)) + 1e-8)
                IoU = iou_pred/len(dataloader.dataset)
        
                train_loss.append(np.array(epoch_loss.detach().cpu())) if phase == 'train' \
                    else valid_loss.append(np.array(epoch_loss.detach().cpu()))
                train_acc.append(np.array(epoch_acc2)) if phase == 'train' \
                    else valid_acc.append((np.array(epoch_acc2)))
                train_dise.append(np.array(epoch_dise)) if phase == 'train' \
                    else valid_dise.append((np.array(epoch_dise)))
                train_dise2.append(np.array(epoch_dise2)) if phase == 'train' \
                    else valid_dise2.append((np.array(epoch_dise2)))
        
                train_IoU.append(np.array(IoU)) if phase == 'train' \
                    else valid_IoU.append((np.array(IoU)))
        
                train_F1score.append(np.array(F1score)) if phase == 'train' \
                    else valid_F1score.append((np.array(F1score)))
        
                train_Spec.append(np.array(Spec)) if phase == 'train' \
                    else valid_Spec.append((np.array(Spec)))
                train_Sens.append(np.array(Sens)) if phase == 'train' \
                    else valid_Sens.append((np.array(Sens)))
                train_Prec.append(np.array(Prec)) if phase == 'train' \
                    else valid_Prec.append((np.array(Prec)))
        
        
        time_elapsed = time.time() - start
        with open(training_tsx, "a") as f:
        # print(model, file=f)     
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60), file=f)
            
        

        ############################
        with open(training_tsx, "a") as f:
                
            print('-' * 10, file=f)
            print('Test Results', file=f)
            print('test_acc', valid_acc, file=f)
            print('test_dise',valid_dise, file=f)
            print('test_IoU',valid_IoU, file=f)
      
                    
        f.close()
        
        


    
    
