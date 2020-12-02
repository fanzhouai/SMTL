"""TODO: docstring
"""


import pathlib
import os
from os.path import join
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms.transforms import *

from torchvision import utils
from PIL import Image
import random
import matplotlib.pyplot as plt
import collections
from collections import Counter

from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


from torchvision import transforms
from torch.utils.data import DataLoader




def join_path(*a):
    return os.path.join(*a)



class BaseImageDataset(Dataset):
    """
    base image dataset
    for image dataset, ``__getitem__`` usually reads an image from a given file path
    the image is guaranteed to be in **RGB** mode
    subclasses should fill ``datas`` and ``labels`` as they need.
    """

    def __init__(self, transform=None, return_id=False):
        self.return_id = return_id
        self.transform = transform or (lambda x : x)
        self.datas = []
        self.labels = []

    def __getitem__(self, index):
        
        if index< len(self.datas):
            idx = index
            im = Image.open(self.datas[index]).convert('RGB')
            im = self.transform(im)
        else:
            re_index = index % len(self.datas)
            idx = re_index
            im = Image.open(self.datas[re_index]).convert('RGB')
            im = self.transform(im)    

        if not self.return_id:
            return im, self.labels[idx]
        return im, self.labels[idx], idx

    def __len__(self):
        return len(self.datas)

class FileListDataset(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :
    image_path label_id
    image_path label_id
    ......
    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(FileListDataset, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line: # avoid empty lines
                    ans = line.split()
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[-1]
                        file = line[:-len(label)].strip()
                        data.append([file, label])
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data] 
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)
        #print()

        self.num_classes = num_classes or max(self.labels) + 1

        def return_datas(self):
            
            return self.datas       
            

def get_dloader_pacs(data_name, re_weighting, path_prefix ,ratio,configure):
    
    source_dataset, source_dataset_test, source_dataset_validation = [], [] , []
    train_loader, test_loader = [], []
    batch_size = configure[data_name]['batch_size']
    test_batch_size = configure[data_name]['test_batch_size']


    for tsk in configure[data_name]['task_list']:
        print('LLLLLLLLLLLLLLL loading the current task '+tsk)
        list_train = configure[data_name]['data_list']['train'][tsk]
        list_test = configure[data_name]['data_list']['test'][tsk]

        need_balance = re_weighting

        source_dataset.append(partial_dataset_pacs(list_path= list_train, path_prefix=path_prefix,
                                        transform=configure[data_name]['train_transform'], num_classes = configure[data_name]['num_classes'],filter=(lambda x: x in range(configure[data_name]['num_classes'])),
                                        num_ratio = ratio))
        source_dataset_validation.append(partial_dataset_pacs(list_path= list_test, path_prefix=path_prefix,
                                        transform=configure[data_name]['train_transform'], num_classes = configure[data_name]['num_classes'],filter=(lambda x: x in range(configure[data_name]['num_classes'])),
                                        num_ratio = 1.0)) 
        source_dataset_test.append(partial_dataset_pacs(list_path= list_test, path_prefix=path_prefix,
                                        transform=configure[data_name]['test_transform'], num_classes = configure[data_name]['num_classes'],filter=(lambda x: x in range(configure[data_name]['num_classes'])),
                                        num_ratio = 1.0)) # we test on the full dataset

    if re_weighting:
        for i in range(len(source_dataset)):
            source_classes = source_dataset[i].labels
            source_freq = Counter(source_classes)
            source_class_weight = {x : 1.0 / source_freq[x] if need_balance else 1.0 for x in source_freq}
            source_weights = [source_class_weight[x] for x in source_dataset[i].labels]
            source_sampler = WeightedRandomSampler(source_weights, len(source_dataset[i].labels))
            train_loader.append(DataLoader(source_dataset[i],batch_size=batch_size, sampler = source_sampler, drop_last=True, num_workers=8))
    else:
        train_loader = [DataLoader(source_dataset[t], batch_size=configure[data_name]['batch_size'], shuffle=True, num_workers=8, drop_last=True)
                    for t in range(len(source_dataset_test))]
    
    val_loader = [DataLoader(source_dataset_validation[t], batch_size=16, shuffle=False, num_workers=0) for t in range(len(source_dataset_test))]
    test_loader = [DataLoader(source_dataset_test[t], batch_size=16, shuffle=False, num_workers=0) for t in range(len(source_dataset_test))]

    print('Counter of labels', Counter(source_dataset[0].labels))
    return train_loader,val_loader, test_loader



class partial_dataset_drift(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :
    image_path label_id
    image_path label_id
    ......
    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', num_ratio=0.1, ratio_drift = None, no_task = 0,
                     transform=None, return_id=False, num_classes=None, data_name = None,filter=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        :ratio_drift -> we are loading which task, for example, in office31 amazon is the first task, dslr is second and webcam is third
        """
        super(partial_dataset_drift, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        self.num_ratio = num_ratio
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line: # avoid empty lines
                    ans = line.split()
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[-1]
                        file = line[:-len(label)].strip()
                        data.append([file, label])
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)
        #print('len self datas', len(self.datas))
        #print('len self labels', len(self.labels))
        

        self.num_classes = num_classes or max(self.labels)+1
        #print('self numclasses', self.num_classes)

        self.group_dict = self.group_datas()
        if num_ratio:

            if not ratio_drift:
                print(' WE ONLY PICK UP PARTIAL DATA')
                self.datas, self.labels = self.get_partial_dataset()

            if ratio_drift:
                print(' WE SIMULATE A LABEL SPACE DRIFT SITUATION')
                self.datas, self.labels = self.get_dataset_with_label_drift(data_name = data_name, no_task = no_task ,
                                                                            ratio_drift = ratio_drift )


            print('Stats of the labels', Counter(self.labels))
        

    def group_datas(self):
        group_dict ={}
        for i in range(len(self.datas)):
            if str(self.labels[i]) not in group_dict:
                group_dict.update({str(self.labels[i]):[self.datas[i]]})
            else:
                group_dict[str(self.labels[i])].append(self.datas[i])
    
        return group_dict



    def get_partial_dataset(self):

        # Return dataset with 
        new_group_dict = {}
        labels = []
        datas = []
        for k in range(self.num_classes):
            i = k
            choice_num = int(self.num_ratio*len(self.group_dict[str(i)]))+1
            np.random.shuffle(self.group_dict[str(i)])
            new_group_dict.update({str(i):self.group_dict[str(i)][:choice_num]})
        
        for k in range(self.num_classes):
            i = k
            for j in range(len(new_group_dict[str(i)])):
                labels.append(int(i))
                datas.append(new_group_dict[str(i)][j])
        return datas, labels

    
    def get_dataset_with_label_drift(self, data_name, no_task, ratio_drift = None):
        #random.seed(42)
        #np.random.seed(42)
        #group_dict = self.group_datas()
        new_group_dict = {}
        
        # Firstly define how many classes to drift for different dataset
        if data_name =='office31':
            drifted_classes = 10
        elif data_name == 'office_home':
            drifted_classes = 16

        labels = []
        datas = []
        for k in range(self.num_classes):
            i = k
            # For task #.no_task, we only drift certain tasks among them
            if ((k>=int(no_task*drifted_classes)) and (k<= (no_task+1)*drifted_classes)):
                #choice_num = int(self.num_ratio*len(self.group_dict[str(i)]))+1
                choice_num =  int(self.num_ratio*(1.0-ratio_drift)*len(self.group_dict[str(i)]))
                if choice_num <1:
                    choice_num = choice_num+1  # Make sure we won't have empty  classes
                np.random.shuffle(self.group_dict[str(i)])
                new_group_dict.update({str(i):self.group_dict[str(i)][:choice_num]})
            else:
                choice_num = int(self.num_ratio*len(self.group_dict[str(i)]))
                np.random.shuffle(self.group_dict[str(i)])
                new_group_dict.update({str(i):self.group_dict[str(i)][:choice_num]})

        for k in range(self.num_classes):
            i = k
            for j in range(len(new_group_dict[str(i)])):

                labels.append(int(i))
                datas.append(new_group_dict[str(i)][j])
                
        return datas, labels
        
        
    def return_datas(self):
            
        return self.datas, self.labels

def get_dloader_office31(data_name, re_weighting, path_prefix,ratio,configure, ratio_drift=None ):
    
    source_dataset, source_dataset_test, source_dataset_validation = [], [] , []
    train_loader, test_loader = [], []
    batch_size = configure[data_name]['batch_size']
    test_batch_size = configure[data_name]['test_batch_size']
    i = 0


    for tsk in configure[data_name]['task_list']:
        print('LLLLLLLLLLLLLLL loading the current task '+tsk)
        list_train = configure[data_name]['data_list']['train'][tsk]
        list_test = configure[data_name]['data_list']['test'][tsk]

        need_balance = re_weighting

        # We only apply data distribution drift on the train dataset while keep the original test dataset

        source_dataset.append(partial_dataset_drift(list_path= list_train, path_prefix=path_prefix,
                                        transform=configure[data_name]['train_transform'], filter=(lambda x: x in range(configure[data_name]['num_classes'])),
                                        num_ratio = ratio, data_name = data_name, num_classes = configure[data_name]['num_classes'],no_task = i, ratio_drift = ratio_drift))
        
        source_dataset_validation.append(partial_dataset(list_path= list_test, path_prefix=path_prefix,
                                        transform=configure[data_name]['train_transform'], num_classes = configure[data_name]['num_classes'],filter=(lambda x: x in range(configure[data_name]['num_classes'])),
                                        num_ratio = 1.0)) 
                    
        source_dataset_test.append(partial_dataset(list_path= list_test, path_prefix=path_prefix,
                                        transform=configure[data_name]['test_transform'], num_classes = configure[data_name]['num_classes'],filter=(lambda x: x in range(configure[data_name]['num_classes'])),
                                        num_ratio = 1.0)) 
        print('number of class now',len(Counter(source_dataset[i].labels)))
        i+=1 # i is used to track which task is loading, used for data drift and data shift setting

    if re_weighting:
        for i in range(len(source_dataset)):
            source_classes = source_dataset[i].labels
            source_freq = Counter(source_classes)
            source_class_weight = {x : 1.0 / source_freq[x] if need_balance else 1.0 for x in source_freq}
            source_weights = [source_class_weight[x] for x in source_dataset[i].labels]
            source_sampler = WeightedRandomSampler(source_weights, len(source_dataset[i].labels))
            train_loader.append(DataLoader(source_dataset[i],batch_size=batch_size, sampler = source_sampler, drop_last=True, num_workers=8))
    else:
        train_loader = [DataLoader(source_dataset[t], batch_size=configure[data_name]['batch_size'], shuffle=True, num_workers=8, drop_last=True)
                    for t in range(len(source_dataset_test))]
    val_loader = [DataLoader(source_dataset_validation[t], batch_size=configure[data_name]['batch_size'], shuffle=True, num_workers=8, drop_last=True)
                    for t in range(len(source_dataset_test))]
    test_loader = [DataLoader(source_dataset_test[t], batch_size=test_batch_size, shuffle=False, num_workers=0) for t in range(len(source_dataset_test))]


    return train_loader, val_loader, test_loader


class partial_dataset(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :
    image_path label_id
    image_path label_id
    ......
    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', num_ratio=0.1, 
                     transform=None, return_id=False, num_classes=None, filter=None, pacs = False):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(partial_dataset, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        self.num_ratio = num_ratio
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line: # avoid empty lines
                    ans = line.split()
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[-1]
                        file = line[:-len(label)].strip()
                        data.append([file, label])
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            if pacs:
                try:
                    self.labels = [int(x[1])-1 for x in data] 
                except ValueError as e:
                    print('invalid label number, maybe there is a space in the image path?')
                    raise e
            else:
                try:
                    self.labels = [int(x[1]) for x in data]
                except ValueError as e:
                    print('invalid label number, maybe there is a space in the image path?')
                    raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)
        #print('len self datas', len(self.datas))
        #print('len self labels', len(self.labels))


        self.num_classes = num_classes or max(self.labels)
        #print('self numclasses', self.num_classes)
        if pacs:
            self.group_dict = self.group_datas_pacs()
        else:
            self.group_dict = self.group_datas()
        if num_ratio:
            self.datas, self.labels = self.get_partial_dataset(pacs = pacs)

    def group_datas(self):
        group_dict ={}
        for i in range(len(self.datas)):
            if str(self.labels[i]) not in group_dict:
                group_dict.update({str(self.labels[i]):[self.datas[i]]})
            else:
                group_dict[str(self.labels[i])].append(self.datas[i])
        #print('group dict is', group_dict)
        #print('len group dict is',len(group_dict))
        return group_dict
    def group_datas_pacs(self):
        group_dict ={}
        for i in range(len(self.datas)):
            if str(self.labels[i]+1) not in group_dict:
                group_dict.update({str(self.labels[i]+1):[self.datas[i]]})
            else:
                group_dict[str(self.labels[i]+1)].append(self.datas[i])
        print('len of group dict', len(group_dict))
        return group_dict
    def get_partial_dataset(self,pacs):
        # Return dataset with 
        new_group_dict = {}
        labels = []
        datas = []
        for k in range(self.num_classes):
            if pacs: 
                i = k+1
            else:
                i = k
            choice_num = int(self.num_ratio*len(self.group_dict[str(i)]))
            #if choice_num<1:
            #    choice_num = choice_num+1 # Make sure we don't have empty c
            np.random.shuffle(self.group_dict[str(i)])
            new_group_dict.update({str(i):self.group_dict[str(i)][:choice_num]})
        
        for k in range(self.num_classes):
            if pacs: 
                i = k+1
            else:
                i = k
            for j in range(len(new_group_dict[str(i)])):
                labels.append(int(i))
                datas.append(new_group_dict[str(i)][j])
        return datas, labels           
        
    def return_datas(self):
            
        return self.datas

class partial_dataset_pacs(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :
    image_path label_id
    image_path label_id
    ......
    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', num_ratio=0.1, 
                     transform=None, return_id=False, num_classes=None, filter=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(partial_dataset_pacs, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        self.num_ratio = num_ratio
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line: # avoid empty lines
                    ans = line.split()
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[-1]
                        file = line[:-len(label)].strip()
                        data.append([file, label])
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1])-1 for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)


        self.num_classes = num_classes or max(self.labels)
        self.group_dict = self.group_datas()
        if num_ratio:
            self.datas, self.labels = self.get_partial_dataset()

    def group_datas(self):
        group_dict ={}
        for i in range(len(self.datas)):
            if str(self.labels[i]) not in group_dict:
                group_dict.update({str(self.labels[i]):[self.datas[i]]})
            else:
                group_dict[str(self.labels[i])].append(self.datas[i])


        return group_dict
    
    def get_partial_dataset(self):
        # Return dataset with 
        new_group_dict = {}
        labels = []
        datas = []
        for k in range(self.num_classes):
            i = k
            choice_num = int(self.num_ratio*len(self.group_dict[str(i)]))+1
            np.random.shuffle(self.group_dict[str(i)])
            new_group_dict.update({str(i):self.group_dict[str(i)][:choice_num]})
        
        for k in range(self.num_classes):
            i = k
            for j in range(len(new_group_dict[str(i)])):
                labels.append(int(i))
                datas.append(new_group_dict[str(i)][j])

        return datas, labels
        
    def return_datas(self):
            
        return self.datas



def get_dloader_office_home(data_name, re_weighting, path_prefix, ratio,configure, ratio_drift=None,):
    
    source_dataset, source_dataset_test, source_dataset_validation = [], [] , []
    train_loader, test_loader = [], []
    batch_size = configure[data_name]['batch_size']
    test_batch_size = configure[data_name]['test_batch_size']

    need_balance = re_weighting
    ratio = str(int(ratio*100))
    i = 0

    for tsk in configure[data_name]['task_list']:
        print('LLLLLLLLLLLLLLL loading the current task'+tsk)
        list_train = configure[data_name]['data_list']['train'][tsk][ratio]
        list_test = configure[data_name]['data_list']['test'][tsk][ratio]

        source_dataset.append(partial_dataset_drift_for_office_home(list_path= list_train, path_prefix='',
                                        transform=configure[data_name]['train_transform'], filter=(lambda x: x in range(configure[data_name]['num_classes'])),
                                         data_name = data_name, num_classes = configure[data_name]['num_classes'], 
                                        no_task = i, ratio_drift = ratio_drift))
        
        source_dataset_validation.append(FileListDataset(list_path= list_test, path_prefix='',
                                        transform=configure[data_name]['train_transform'], filter=(lambda x: x in range(configure[data_name]['num_classes']))))
                                        # We test on the full dataset
        source_dataset_test.append(FileListDataset(list_path= list_test, path_prefix='',
                                        transform=configure[data_name]['test_transform'], filter=(lambda x: x in range(configure[data_name]['num_classes']))))
                                        # We test on the full dataset
        i+=1


    if re_weighting:
        need_balance = re_weighting
        for i in range(len(source_dataset)):
            source_classes = source_dataset[i].labels
            source_freq = Counter(source_classes)
            source_class_weight = {x : 1.0 / source_freq[x] if need_balance else 1.0 for x in source_freq}
            source_weights = [source_class_weight[x] for x in source_dataset[i].labels]
            source_sampler = WeightedRandomSampler(source_weights, len(source_dataset[i].labels))
            train_loader.append(DataLoader(source_dataset[i],batch_size=batch_size, sampler = source_sampler, drop_last=True, num_workers=8))
    else:
        train_loader = [DataLoader(source_dataset[t], batch_size=configure[data_name]['batch_size'], shuffle=True, num_workers=8, drop_last=True)
                    for t in range(len(source_dataset_test))]

    val_loader = [DataLoader(source_dataset_validation[t], batch_size=configure[data_name]['batch_size'], shuffle=True, num_workers=8, drop_last=True)
                    for t in range(len(source_dataset_validation))]
    test_loader = [DataLoader(source_dataset_test[t], batch_size=16, shuffle=False, num_workers=0) for t in range(len(source_dataset_test))]
        
    return train_loader,val_loader, test_loader



class partial_dataset_drift_for_office_home(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :
    image_path label_id
    image_path label_id
    ......
    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', ratio_drift = None, no_task = 0,
                     transform=None, return_id=False, num_classes=None, data_name = None,filter=None, shift_num = None, rd_seed = 0):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        :ratio_drift -> we are loading which task, for example, in office31 amazon is the first task, dslr is second and webcam is third
        """
        super(partial_dataset_drift_for_office_home, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        #self.num_ratio = num_ratio
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                if line: # avoid empty lines
                    ans = line.split()
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[-1]
                        file = line[:-len(label)].strip()
                        data.append([file, label])
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)
        #print('len self datas', len(self.datas))
        #print('len self labels', len(self.labels))
        

        self.num_classes = num_classes or max(self.labels)+1
        #print('self numclasses', self.num_classes)
        self.group_dict = self.group_datas()
   

        if not ratio_drift:
            print(' WE ONLY PICK UP PARTIAL DATA ')
                #self.datas, self.labels = self.get_partial_dataset()

        if ratio_drift:
            print(' WE SIMULATE A LABEL SPACE DRIFT SITUATION')
            self.datas, self.labels = self.get_dataset_with_label_drift(data_name = data_name, no_task = no_task ,
                                                                            ratio_drift = ratio_drift )

                #print('labels', self.labels)


    def group_datas(self):
        group_dict ={}
        for i in range(len(self.datas)):
            if str(self.labels[i]) not in group_dict:
                group_dict.update({str(self.labels[i]):[self.datas[i]]})
            else:
                group_dict[str(self.labels[i])].append(self.datas[i])


        return group_dict
    
    def get_dataset_with_label_drift(self, data_name, no_task, ratio_drift = None):
        #random.seed(42)
        #np.random.seed(42)
        #group_dict = self.group_datas()
        new_group_dict = {}
        
        # Firstly define how many classes to drift for different dataset
        if data_name =='office31':
            drifted_classes = 10
        elif data_name == 'office_home':
            drifted_classes = 16
        elif data_name == 'digits' or 'office_caltech':
            drifted_classes = 3
        elif data_name == 'PACS' or 'pacs':
            drifted_classes = 2

        labels = []
        datas = []
        print('The counter before drift', Counter(self.labels))
        for k in range(self.num_classes):
            i = k
            # For task #.no_task, we only drift certain tasks among them
            if ((k>=int(no_task*drifted_classes)) and (k<= (no_task+1)*drifted_classes)):
                #choice_num = int(self.num_ratio*len(self.group_dict[str(i)]))+1
                choice_num =  int((1.0-ratio_drift)*len(self.group_dict[str(i)]))
                if choice_num <1:
                    choice_num = choice_num+1  # Make sure we won't have empty  classes
                np.random.shuffle(self.group_dict[str(i)])
                new_group_dict.update({str(i):self.group_dict[str(i)][:choice_num]})
            else:
                #choice_num = int(len(self.group_dict[str(i)]))
                np.random.shuffle(self.group_dict[str(i)])
                new_group_dict.update({str(i):self.group_dict[str(i)]})

        for k in range(self.num_classes):
            i = k
            for j in range(len(new_group_dict[str(i)])):

                labels.append(int(i))
                datas.append(new_group_dict[str(i)][j])
        print('The counter after drift simulation', Counter(labels))
        return datas, labels
        
        
    def return_datas(self):
            
        return self.datas, self.labels

def get_dloader_office_caltech(data_name, re_weighting, ratio, configure):
    source_dataset, source_dataset_test, source_dataset_validation = [], [], []
    train_loader, val_loader, test_loader = [], [], []
    batch_size = configure[data_name]['batch_size']
    test_batch_size = configure[data_name]['test_batch_size']
    ratio = str(int(ratio*100))
    need_balance = re_weighting

    for tsk in configure[data_name]['task_list']:
        list_train = configure[data_name]['data_list']['train'][tsk][ratio]
        list_test = configure[data_name]['data_list']['test'][tsk][ratio]

        source_dataset.append(FileListDataset(list_path=list_train, path_prefix='',
                                              transform=configure[data_name]['train_transform'],
                                              num_classes=configure[data_name]['num_classes'],
                                              filter=(lambda x: x in range(configure[data_name]['num_classes']))))
        source_dataset_validation.append(FileListDataset(list_path=list_test, path_prefix='',
                                                   transform=configure[data_name]['train_transform'],
                                                   num_classes=configure[data_name]['num_classes'],
                                                   filter=(lambda x: x in range(configure[data_name]['num_classes']))))
        source_dataset_test.append(FileListDataset(list_path=list_test, path_prefix='',
                                                   transform=configure[data_name]['test_transform'],
                                                   num_classes=configure[data_name]['num_classes'],
                                                   filter=(lambda x: x in range(configure[data_name]['num_classes']))))
    if re_weighting:
        for i in range(len(source_dataset)):
            source_classes = source_dataset[i].labels
            source_freq = Counter(source_classes)
            source_class_weight = {x: 1.0 / source_freq[x] if need_balance else 1.0 for x in source_freq}
            source_weights = [source_class_weight[x] for x in source_dataset[i].labels]
            source_sampler = WeightedRandomSampler(source_weights, len(source_dataset[i].labels))
            train_loader.append(
                DataLoader(source_dataset[i], batch_size=batch_size, sampler=source_sampler, drop_last=True,
                           num_workers=8))
            test_loader.append(
                DataLoader(source_dataset_test[i], batch_size=test_batch_size, shuffle=True, num_workers=8))
    else:
        train_loader = [
            DataLoader(source_dataset[t], batch_size=configure[data_name]['batch_size'], shuffle=True, num_workers=8,
                       drop_last=True) for t in range(len(source_dataset_train))]

    val_loader =  [DataLoader(source_dataset[t], batch_size=configure[data_name]['batch_size'], shuffle=True, num_workers=8,
                       drop_last=True) for t in range(len(source_dataset_test))]

    test_loader = [DataLoader(source_dataset_test[t], batch_size=test_batch_size, shuffle=True, num_workers=8) for t in
                       range(len(source_dataset_test))]

    return train_loader, val_loader, test_loader