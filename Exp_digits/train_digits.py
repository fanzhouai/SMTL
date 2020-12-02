import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn

import util


import data_loading as db
from tqdm import tqdm

import time
from method_digits import MTL_Semantic





parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int)

parser.add_argument('--initial_lr', default=1e-3, type=float)                # initial lr

parser.add_argument('--down_period', default=5, type=int) 
parser.add_argument('--lr_decy', default=0.95, type=float) 

parser.add_argument('--max_epoch', default=100, type=int) 
parser.add_argument('--train_samples', default=3000, type=int) 


args = parser.parse_args()

print('args for the experiments', args)


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = torch.cuda.is_available()


max_epoch = args.max_epoch


args_train = {'img_size': 28,
        'lr': args.initial_lr,
        'tr_smpl': args.train_samples,
        'test_smpl': 10000,
        'task_list': ['mnist', 'svhn', 'm_mnist'],
        'num_classes': 10,
        'c3':0.5}


train_loader, test_loader, validation_loader = db.data_loading(args_train['img_size'],  args_train['tr_smpl'],args_train['test_smpl'], args_train['task_list'])



args_train['lr'] = args.initial_lr
args_train['down_period'] = args.down_period
args_train['lr_decay_rate'] = args.lr_decy

args_train['c3'] = 0.5

args_train['ft_extrctor_prp'] = {'layer1': {'conv': [1, 32, 5, 1, 2], 'elu': [], 'maxpool': [3, 2, 0]},
                               'layer2': {'conv': [32, 64, 5, 1, 2], 'elu': [], 'maxpool': [3, 2, 0]}}

args_train['hypoth_prp'] = {
                'layer3': {'fc': [util.in_feature_size(args_train['ft_extrctor_prp'], args_train['img_size']), 128], 'act_fn': 'elu'},
                'layer4': {'fc': [128, 10], 'act_fn': 'softmax'}}

MTL_algo = MTL_Semantic( train_loader = train_loader, test_loader = test_loader, val_loader = validation_loader, args = args_train)


best_acc = 0
time_series = []
for epoch in range(max_epoch):

    start = time.time()
    total_loss = MTL_algo.model_fit(epoch = epoch)
    end = time.time()
    time_round_round = end - start
    if epoch >0:
        time_series.append(time_round_round)
    print('Training time for one round', time_round_round)

    tasks_trAcc = MTL_algo.model_eval(epoch = epoch,mode ='train')
    tasks_valAcc = MTL_algo.model_eval(epoch = epoch,mode ='val')
    tasks_teAcc = MTL_algo.model_eval(epoch = epoch, mode = 'test')

    if np.mean(best_acc)< np.mean((tasks_teAcc).tolist()):
        best_acc = (tasks_teAcc).tolist()

    print('\t --------- The average acc is ', np.mean((tasks_teAcc).tolist()))

print('The best acc is',best_acc)
print('Average of the acc is',np.mean(best_acc))

print('The average time for one training round',np.mean(time_series))
print('------------FINISH------------------')
del MTL_algo