"""
    https://arxiv.org/pdf/2407.10157

    https://github.com/usagisukisuki/Adaptive_t-vMF_Dice_loss

        impact factor=7.7

        kappa = is a concentration parameter that adjusts the shape of the similarity function

"""

import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F

##### tvMF Dice loss #####
class tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes, kappa=None):
        super(tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.kappa = kappa

    ### one-hot encoding ###
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    ### tvmf dice loss ###
    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        smooth = 1.0

        score = F.normalize(score, p=2, dim=[0,1,2])
        target = F.normalize(target, p=2, dim=[0,1,2])
        cosine = torch.sum(score * target)
        intersect =  (1. + cosine).div(1. + (1.- cosine).mul(kappa)) - 1.
        loss = (1 - intersect)**2.0

        return loss

    ### main ###
    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0

        for i in range(0, self.n_classes):
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], self.kappa)
            loss += tvmf_dice
        return loss / self.n_classes


##### Adaptive tvMF Dice loss #####
class Adaptive_tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(Adaptive_tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes

    ### one-hot encoding ###
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    ### tvmf dice loss ###
    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        smooth = 1.0

        score = F.normalize(score, p=2, dim=[0,1,2])
        target = F.normalize(target, p=2, dim=[0,1,2])
        cosine = torch.sum(score * target)
        intersect =  (1. + cosine).div(1. + (1.- cosine).mul(kappa)) - 1.
        loss = (1 - intersect)**2.0

        return loss

    ### main ###
    def forward(self, inputs, target, kappa=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0

        for i in range(0, self.n_classes):
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], kappa[i])
            loss += tvmf_dice
        return loss / self.n_classes
    


class DiceScoreCoefficient(nn.Module):
    def __init__(self, n_classes):
        super(DiceScoreCoefficient, self).__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def fast_hist(self, label_true, label_pred, labels):
        mask = (label_true >= 0) & (label_true < labels)
        hist = np.bincount(labels * label_true[mask].astype(int) + label_pred[mask], minlength=labels ** 2,
        ).reshape(labels, labels)
        return hist

    def _dsc(self, mat):
        diag_all = np.sum(np.diag(mat))
        fp_all = mat.sum(axis=1)
        fn_all = mat.sum(axis=0)
        tp_tn = np.diag(mat)
        precision = np.zeros((self.n_classes)).astype(np.float32)
        recall = np.zeros((self.n_classes)).astype(np.float32)    
        f2 = np.zeros((self.n_classes)).astype(np.float32)

        for i in range(self.n_classes):
            if (fp_all[i] != 0)and(fn_all[i] != 0):   
                precision[i] = float(tp_tn[i]) / float(fp_all[i])
                recall[i] = float(tp_tn[i]) / float(fn_all[i])
                if (precision[i] != 0)and(recall[i] != 0):  
                     f2[i] = (2.0*precision[i]*recall[i]) / (precision[i]+recall[i])
                else:       
                    f2[i] = 0.0
            else:
                precision[i] = 0.0
                recall[i] = 0.0

        return f2


    ### main ###
    def forward(self, output, target):
        output = np.array(output)
        target = np.array(target)
        seg = np.argmax(output,axis=1)

        for lt, lp in zip(target, seg):
            self.confusion_matrix += self.fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

        dsc = self._dsc(self.confusion_matrix)

        return dsc