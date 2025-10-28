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


from medpy.metric import dc
from scipy.ndimage import zoom

# from utils.loss import tvMF_DiceLoss, Adaptive_tvMF_DiceLoss, DiceScoreCoefficient


def run_val(net,valloader,args,Best_dcs,best_epoch_num):
    logging.info("Validation ===>")
    dc_sum=0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]

        val_image_batch = val_image_batch.squeeze(0).cpu().detach().numpy()
        val_label_batch = val_label_batch.squeeze(0).cpu().detach().numpy()

        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y), order=3) # not for double_maxvits
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()
        
        val_outputs = net(val_image_batch)
        #print(len(P))
        
        val_outputs = torch.softmax(val_outputs, dim=1)

        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()
        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        else:
            val_outputs = val_outputs

        dc_sum+=dc(val_outputs,val_label_batch[:])
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f [%f]' % (performance, Best_dcs, best_epoch_num))

    print('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))
    #print("val avg_dsc: %f" % (performance))
    return performance


def trainer_acdc(args, model, snapshot_path):

    # adjust the parameter kappa #
    def adjust_kappa(mm):
        return torch.Tensor(mm*32.0).cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # print("[INFO] CoAMamba trainer")
    
    from utils.dataset_acdc_seg import ACDCdataset as ACDCdataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # weight_decay = 1e-4  # update from 0.0001

    db_train = ACDCdataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])])
                            )
    print("Dataset folder at: {}" .format(args.root_path) )
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True
                             , pin_memory=True
                             , num_workers=1
                             , worker_init_fn=worker_init_fn
                             )
    
    db_val=ACDCdataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid")
    valloader=DataLoader(db_val, batch_size=1, shuffle=False)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)


    optimizer = optim.AdamW(model.parameters(), lr=base_lr * 0.01, betas=(0.5, 0.99))

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(trainloader), T_mult=2)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    Best_dcs = 0.80
    best_epoch_num = 0
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


        #### run to select the best model
        avg_dcs = run_val(model,valloader,args,Best_dcs,best_epoch_num)
        if avg_dcs > Best_dcs:
            save_model_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_model_path)
            logging.info("save model to {} at eps {}".format(save_model_path,epoch_num))
            print("save model to {}".format(save_model_path))
            best_epoch_num = epoch_num
            Best_dcs = avg_dcs

        # if epoch_num in [60,100,120,150]:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"