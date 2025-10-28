import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_btcv_seg_v1 import BTCV_dataset as Synapse_dataset, RandomGenerator
from utils.evals import test_single_volume


parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,default='MambaCAFU_v2', choices=['MambaCAFU_v1','MambaCAFU_v2']) 

parser.add_argument('--volume_path', type=str,
                    default='data_sets/Synapse/dataset13classes/test_vol_h5_new', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=14, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='data_sets/Synapse/lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--test_save_dir', type=str, default='predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')

parser.add_argument('--seed', type=int, default=2222, help='random seed')

args = parser.parse_args()

if(args.num_classes == 14):
    classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'portal vein and splenic vein', 'pancreas', 'right adrenal gland', 'left adrenal gland']
else:
    classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    args.exp = args.model_name

    if args.exp == 'MambaCAFU_v1':
        import  networks.network as MyPYNet
        net  = MyPYNet.MambaCAFU(in_channels=3, num_classes= args.num_classes, version = 'v0').cuda()
        snapshot_path = "./model/ACDC/"+args.exp
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    elif args.exp == 'MambaCAFU_v2':
        import  networks.network as MyPYNet
        net  = MyPYNet.MambaCAFU(in_channels=3, num_classes= args.num_classes, version = 'v2').cuda()
        snapshot_path = "./model/ACDC/"+args.exp
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)  

    list_epoch_num = ["120"]
    for epoch_num in list_epoch_num:
        weight_name = f"epoch_{epoch_num}.pth"
        snapshot = os.path.join(snapshot_path, weight_name) #'epoch_100.pth'
        if not os.path.exists(snapshot): 
            snapshot = snapshot.replace('best_model', str(args.max_epochs-1))
            if not os.path.exists(snapshot): 
                continue
        print(f"Model weights at {snapshot}")
        args.model_weights = snapshot
        net.load_state_dict(torch.load(snapshot))
        snapshot_name = snapshot_path.split('/')[-1]

        log_folder = 'test_log/test_log_' + args.exp
        os.makedirs(log_folder, exist_ok=True)
        logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(snapshot_name)

        if args.is_savenii:
            args.test_save_dir = 'predictions'
            test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
            os.makedirs(test_save_path, exist_ok=True)
        else:
            test_save_path = None
        inference(args, net, test_save_path)