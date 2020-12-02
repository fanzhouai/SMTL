import os
import json
from collections import defaultdict
import numpy as np

#from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.optim as optim
from torch.utils.data import DataLoader


import torchvision
from torchvision import transforms


import util
from alpha_opt import*
from torch.optim import lr_scheduler



class MTL_Semantic():
    def __init__(self, train_loader, test_loader, val_loader, args):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.args = args
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



        self.decay = 0.3        
        self.num_tsk = len(self.args['task_list'])
        self.n_class = self.args['num_classes']

        self.num_tasks = self.num_tsk
        self.centroids = torch.zeros(self.num_tasks,self.n_class, 128) # Centroids size for digits experiments is 128

        self.CEloss, self.MSEloss, self.BCEloss = nn.CrossEntropyLoss(reduction='none'), nn.MSELoss(reduction='none'), nn.BCEWithLogitsLoss(reduction='mean')
        self.cudable = True
        if self.cudable:
           self.CEloss, self.MSEloss, self.BCEloss = self.CEloss.cuda(), self.MSEloss.cuda(), self.BCEloss.cuda()
           self.centroids = self.centroids.cuda()


        self.num_tsk = len(self.args['task_list'])
        

        self.num_tasks = self.num_tsk

        self.alpha = np.ones((self.num_tsk, self.num_tsk)) * (0.1 / (self.num_tsk - 1))
        np.fill_diagonal(self.alpha, 0.9)

        self.down_period = self.args['down_period']
        self.lr_decay_rate = self.args['lr_decay_rate']

        self.lr = args['lr']
        self.c3_value = args['c3']



        # Constructing F -> H 
        self.FE = util.feature_extractor(args['ft_extrctor_prp']).construct().to(self.device)
        print (self.FE)

        self.hypothesis = [util.classifier(args['hypoth_prp']).to(self.device) for _ in range(self.num_tsk)]
        print (self.hypothesis[0])


    def model_fit(self,epoch):

        
        best_acc = 0

        lamb =0.1

        if ((epoch+1)%self.down_period) ==0 and (self.lr>1e-4) :
            self.lr = self.lr*self.lr_decay_rate

        if epoch>80:
            self.lr = self.lr*0.99
        all_parameters_h = sum([list(h.parameters()) for h in self.hypothesis], [])

        self.optimizer = optim.Adam(list(self.FE.parameters()) + list(all_parameters_h),
                                       lr=self.lr,  weight_decay = 1e-5)


        sem_mtrx = np.zeros((self.num_tsk, self.num_tsk))

        loss_mtrx_hypo_vlue = np.zeros((self.num_tsk, self.num_tsk))
        weigh_loss_hypo_vlue, correct_hypo = np.zeros(self.num_tsk), np.zeros(self.num_tsk)
        Total_loss = 0
        n_batch = 0

        # set train mode
        self.FE.train()
        for t in range(self.num_tsk):
            self.hypothesis[t].train()


        for tasks_batch in zip(*self.train_loader ):
            Loss_1, Loss_2 = 0, 0
            semantic_loss = 0
            n_batch += 1
            # data = (x,y)
            inputs = torch.cat([batch[0] for batch in tasks_batch])


            btch_sz = len(tasks_batch[0][0])
            targets = torch.cat([batch[1] for batch in tasks_batch])

            # inputs = (x1,...,xT)  targets = (y1,...,yT)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            features = self.FE(inputs)
            features = features.view(features.size(0), -1)

            
            for t in range(self.num_tsk):
                w = torch.tensor([np.tile(self.alpha[t, i], reps=len(data[0])) for i, data in enumerate(tasks_batch)],
                                 dtype=torch.float).view(-1)
                w = w.to(self.device)

                label_prob, fea = self.hypothesis[t](features)

                pred = label_prob[t * (btch_sz):(t + 1) * btch_sz].argmax(dim=1, keepdim=True)
                correct_hypo[t] += (
                (pred.eq(targets[t * btch_sz:(t + 1) * btch_sz].view_as(pred)).sum().item()) / btch_sz)

                hypo_loss = torch.mean(w * F.cross_entropy(label_prob, targets, reduction='none'))


                Loss_1 += hypo_loss
                weigh_loss_hypo_vlue[t] += hypo_loss.item()



                for k in range(t + 1, self.num_tsk):
                    alpha_domain = torch.tensor(self.alpha[t, k] + self.alpha[k, t], dtype=torch.float)
                    alpha_domain = alpha_domain.to(self.device)
                    

                    sem_fea_t = fea[t * btch_sz:(t + 1) * btch_sz]
                    sem_fea_k = fea[k * btch_sz:(k + 1) * btch_sz]


                        
                    labels_t = targets[t * btch_sz:(t + 1) * btch_sz]
                    labels_k = targets[k * btch_sz:(k + 1) * btch_sz]

                        


                    _,d = sem_fea_t.shape


                    # image number in each class
                    ones = torch.ones_like(labels_t, dtype=torch.float)
                    zeros = torch.zeros(self.n_class)
                    if self.cudable:
                        zeros = zeros.cuda()
                    # smaples per class
                    t_n_classes = zeros.scatter_add(0, labels_t, ones)
                    k_n_classes = zeros.scatter_add(0, labels_k, ones)

                    # image number cannot be 0, when calculating centroids
                    ones = torch.ones_like(t_n_classes)
                    t_n_classes = torch.max(t_n_classes, ones)
                    k_n_classes = torch.max(k_n_classes, ones)

                    # calculating centroids, sum and divide
                    zeros = torch.zeros(self.n_class, d)
                    if self.cudable:
                        zeros = zeros.cuda()
                    t_sum_feature = zeros.scatter_add(0, torch.transpose(labels_t.repeat(d, 1), 1, 0), sem_fea_t)
                    k_sum_feature = zeros.scatter_add(0, torch.transpose(labels_k.repeat(d, 1), 1, 0), sem_fea_k)
                    current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1))
                    current_k_centroid = torch.div(k_sum_feature, k_n_classes.view(self.n_class, 1))

                    # Moving Centroid
                    decay = self.decay
                    t_centroid = (1-decay) * self.centroids[t] + decay * current_t_centroid
                    k_centroid = (1-decay) * self.centroids[k] + decay * current_k_centroid
                        
                    s_loss = self.MSEloss(t_centroid, k_centroid)
                    semantic_loss += torch.mean(s_loss)



                    self.centroids[t] = t_centroid.detach()
                    self.centroids[k]= k_centroid.detach()



            Loss_2 = torch.mean(alpha_domain* semantic_loss) 
            Loss =  torch.mean(Loss_1)+ lamb * Loss_2* (1.0 / self.num_tsk) 

            self.optimizer.zero_grad()
            Loss.backward(retain_graph=True)
            self.optimizer.step()

        if epoch>0:
            c_2, c_3 = 1 * np.ones(self.num_tsk), self.c3_value * np.ones(self.num_tsk)
            self.alpha = min_alphacvx(self.alpha.T, c_2, c_3, loss_mtrx_hypo_vlue.T, sem_mtrx.T)

            self.alpha = self.alpha.T


        Total_loss += Loss.item()

        return  Total_loss

    def model_eval(self, epoch,mode = 'test'):

        if mode =='train':
            data_loader = self.train_loader
        elif mode =='val':
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        loss_hypo_vlue = np.zeros(self.num_tsk)
        correct_hypo = np.zeros(self.num_tsk)
        self.FE.eval()
        for t in range(self.num_tsk):
            n_batch_t = 0
            self.hypothesis[t].eval()
            #for j in range(t + 1, self.num_tsk):
            #    self.discrm['{}{}'.format(t, j)].eval()

            for inputs, targets in (data_loader[t]):
                n_batch_t += 1
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                features = self.FE(inputs)
                features = features.view(features.size(0), -1)
                label_prob,_ = self.hypothesis[t](features)
                pred = label_prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_hypo[t] += ((pred.eq(targets.view_as(pred)).sum().item()) / len(pred))
                loss_hypo_vlue[t] += F.cross_entropy(label_prob, targets, reduction='mean').item()
                #if n_batch_t % 100 == 0:
                #    grid_img = torchvision.utils.make_grid(inputs, nrow=5, padding=30)
                #    self.writer.add_image('result Image_' + phase, grid_img)

            loss_hypo_vlue[t] /= n_batch_t
            correct_hypo[t] /= n_batch_t


            #self.writer.add_scalars('task_' + str(t) + '/loss', {'loss_' + phase: loss_hypo_vlue[t]}, epoch)
            #self.writer.add_scalars('task_' + str(t) + '/Acc', {'Acc_' + phase: correct_hypo[t]}, epoch)

        print('\t ======== EPOCH:{}'.format(epoch))

        print('\t === hypothesiz ** ' +str(mode)+ ' ** loss \n' + str(loss_hypo_vlue))
        print('\t === hypothesiz ** ' +str(mode)+ ' ** accuracy \n' + str(correct_hypo * 100))
        print('The learning rate is', self.lr)

        return correct_hypo






