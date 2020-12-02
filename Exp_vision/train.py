import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn

from configure import *
from dloader import *
from method import MTL_Semantic

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dataset', default='office31', type=str,
                    help='choose the dataset for training, pacs, office_caltech or office_home')

parser.add_argument('--initial_lr', default=1e-3, type=float)  # initial lr

parser.add_argument('--ratio', default=0.2, type=float)  # experiments ratio

parser.add_argument('--down_period', default=5, type=int)
parser.add_argument('--lr_decy', default=0.95, type=float)


parser.add_argument('--max_epoch', default=180, type=int)


parser.add_argument("--drift_ratio", type=float, help="learning_rate_mtr", default=0.1)
parser.add_argument('--re_weighting', default= False, type=bool)

args = parser.parse_args()

print('args for the experiments', args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = torch.cuda.is_available()

#batch_size = 100
max_epoch = args.max_epoch


if args.dataset =='office31':
    train_loader, val_loader, test_loader = get_dloader_office31(data_name = args.dataset, re_weighting = True, path_prefix = './data/', ratio = args.ratio, configure = config,
                            ratio_drift=args.drift_ratio)
elif args.dataset =='pacs':
    train_loader,val_loader, test_loader = get_dloader_pacs(data_name = args.dataset,re_weighting = True, path_prefix = './data/PACS/kfold/', ratio = args.ratio, configure = config)
elif args.dataset == 'office_home':
    train_loader, val_loader, test_loader = get_dloader_office_home(data_name = args.dataset, re_weighting = True, path_prefix = '.', ratio = args.ratio, configure = config, ratio_drift=args.drift_ratio)
elif args.dataset == 'office_caltech':
    train_loader, val_loader, test_loader = get_dloader_office_caltech(data_name= 'office_caltech', re_weighting = True,ratio = args.ratio, configure = config)

n_class = config[args.dataset]['num_classes']

batch_size = config[args.dataset]['batch_size']
args_train = config[args.dataset]


print('GPU: {}'.format(args.gpu))

args_train['dataset'] = args.dataset
args_train['lr'] = args.initial_lr
args_train['down_period'] = args.down_period
args_train['lr_decay_rate'] = args.lr_decy

args_train['c3'] = 0.5


MTL_algo = MTL_Semantic(train_loader=train_loader, val_loader = val_loader,test_loader=test_loader, args=args_train , dataset = args.dataset)


best_acc = 0
total_loss =[]
for epoch in range(max_epoch):

    total_loss.append(MTL_algo.model_fit(epoch=epoch))
    
    tasks_trAcc = MTL_algo.model_eval(epoch=epoch,mode = 'train')
    tasks_valAcc = MTL_algo.model_eval(epoch=epoch,mode = 'val')
    tasks_teAcc = MTL_algo.model_eval(epoch=epoch,mode = 'test')

    if np.mean(best_acc) < np.mean((tasks_teAcc).tolist()):
        best_acc = (tasks_teAcc).tolist()

    print('\t --------- The average acc is ', np.mean((tasks_teAcc).tolist()))

print('The best acc is', best_acc)
print('Average of the acc is', np.mean(best_acc))
print('+++++++++++++++ FINISH ++++++++++++++')


