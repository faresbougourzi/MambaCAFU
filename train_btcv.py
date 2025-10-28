import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


from experiments.btcv.trainer import trainer_synapse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str,default='MambaCAFU_v2', choices=['MambaCAFU_v1','MambaCAFU_v2']) 

parser.add_argument('--root_path', type=str,
                    default='data_sets/Synapse/dataset13classes/train_npz_new', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='data_sets/Synapse/lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=14, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=120, help='maximum epoch number to train') #
parser.add_argument('--batch_size', type=int,
                    default=18, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')

args = parser.parse_args()

print("[DEBUG][args]=",args)

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

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': 'data_sets/Synapse/dataset13classes/train_npz_new',
            'list_dir': 'data_sets/Synapse/lists/lists_Synapse',
            'num_classes': 14,
        },
    }


    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    args.exp = args.model_name

    if args.exp == 'MambaCAFU_v1':
        import  networks.network as MyPYNet
        net  = MyPYNet.MambaCAFU(in_channels=3, num_classes= args.num_classes, version = 'v0').cuda()
        snapshot_path = "./model/BTCV/"+args.exp
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    elif args.exp == 'MambaCAFU_v2':
        import  networks.network as MyPYNet
        net  = MyPYNet.MambaCAFU(in_channels=3, num_classes= args.num_classes, version = 'v2').cuda()
        snapshot_path = "./model/BTCV/"+args.exp
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)  
            
    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)