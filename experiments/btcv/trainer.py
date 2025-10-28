import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from utils.evals import test_single_volume, DiceLoss

import torch.nn.functional as F
# from utils.loss import tvMF_DiceLoss, Adaptive_tvMF_DiceLoss, DiceScoreCoefficient



def trainer_synapse(args, model, snapshot_path):

    # adjust the parameter kappa #
    def adjust_kappa(mm):
        return torch.Tensor(mm*32.0).cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    
    from utils.dataset_btcv_seg_v1 import BTCV_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # weight_decay = 1e-4  # update from 0.0001

    db_train = BTCV_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])])
                                , nclass=args.num_classes
                            )
    print("Dataset folder at: {}" .format(args.root_path) )
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True
                             , pin_memory=True
                             , num_workers=1
                             , worker_init_fn=worker_init_fn
                             )
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # adaptive_loss = Adaptive_tvMF_DiceLoss(n_classes=num_classes)
    # kappa = torch.Tensor(np.zeros((num_classes))).cuda()
    

    optimizer = optim.AdamW(model.parameters(), lr=base_lr * 0.01, betas=(0.5, 0.99))

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(trainloader), T_mult=2)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()

            outputs = model(image_batch)
            
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            
            loss = 0.4 * loss_ce + 0.6 * loss_dice #FocalUnet paper


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', scheduler.get_last_lr()[0], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('epoch_num %d ,iteration %d : loss : %f, loss_ce: %f' % (epoch_num,iter_num, loss.item(), loss_ce.item()))

        if epoch_num in [120]:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break




    writer.close()
    return "Training Finished!"