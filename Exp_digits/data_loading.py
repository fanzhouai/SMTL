"""TODO: docstring
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import collections
from collections import Counter
import util
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np




def data_loading (img_size, num_tr_smpl,num_test_smpl, tsk_list, re_weighting = False ):
    our_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])
    source_dataset, source_dataset_test, source_dataset_validation = [], [] , []

    train_loader = []
    for tsk in tsk_list:
        print ('LLLLLLLLLLLLLLL loading the current task '+tsk)
        if tsk =='mnist':
            source_dataset.append(util.Local_Dataset_digit(data_name='mnist', set='train', data_path='data/mnist', transform=our_transform,
                                 num_samples=num_tr_smpl))
            source_dataset_test.append(
                util.Local_Dataset_digit(data_name='mnist', set='test', data_path='data/mnist',transform=our_transform,
                                         num_samples=num_test_smpl))
            source_dataset_validation.append(
                util.Local_Dataset_digit(data_name='mnist', set='validation', data_path='data/mnist',
                                         transform=our_transform,
                                         num_samples=1000))
        if tsk == 'm_mnist':

            source_dataset.append(util.Local_Dataset_digit(data_name='m_mnist', set='train', data_path='data/mnist_m',
                                                           transform=our_transform,
                                                           num_samples=num_tr_smpl))
            source_dataset_test.append(
                util.Local_Dataset_digit(data_name='m_mnist', set='test', data_path='data/mnist_m',transform=our_transform,
                                         num_samples=num_test_smpl))
            source_dataset_validation.append(
                util.Local_Dataset_digit(data_name='m_mnist', set='validation', data_path='data/mnist_m',
                                         transform=our_transform,
                                         num_samples=1000))

        if tsk =='svhn':

            source_dataset.append(util.Local_SVHN(root='data/SVHN', split='train', transform=our_transform, download=True,
                            num_smpl=num_tr_smpl))
            source_dataset_test.append(
                util.Local_SVHN(root='data/SVHN', split='test',transform=our_transform, download=True,
                                num_smpl=num_test_smpl))
            source_dataset_validation.append(
                util.Local_SVHN(root='data/SVHN', split='extra', transform=our_transform, download=True,
                                    num_smpl=1000))

    need_balance = re_weighting
    if re_weighting:
        for i in range(len(source_dataset)):
            
            if type(source_dataset[i].targets).__module__ == np.__name__:
                source_classes = source_dataset[i].targets
            else:
                source_classes = source_dataset[i].targets.numpy()
            
            source_freq = Counter(source_classes)
            #print('------------- source freq is',source_freq)
            source_class_weight = {x : 1.0 / source_freq[x] if need_balance else 1.0 for x in source_freq}
            #print('-----------source_class_weight is',source_class_weight)
            source_weights = [source_class_weight[x] for x in source_classes]
            source_sampler = WeightedRandomSampler(source_weights, len(source_classes))
            #train_loader.append(DataLoader(source_dataset[i],batch_size=batch_size, sampler = source_sampler, drop_last=True, num_workers=8))
            
            train_loader.append(DataLoader(source_dataset[i], batch_size=16, sampler = source_sampler, num_workers=0))
            #[DataLoader(source_dataset[t], batch_size=16, shuffle=True, num_workers=0)
            #        for t in range(len(source_dataset_test))]
            #test_loader.append(DataLoader(source_dataset_test[i],batch_size=test_batch_size, shuffle = True, num_workers=8))
    else:
        train_loader = [DataLoader(source_dataset[t], batch_size=16, shuffle=True, num_workers=0)
                    for t in range(len(source_dataset_test))]

    test_loader = [
        DataLoader(source_dataset_test[t], batch_size=128, shuffle=False, num_workers=0) for t in
        range(len(source_dataset_test))]


    validation_loader = [
        DataLoader(source_dataset_validation[t], batch_size=128, shuffle=False, num_workers=0) for t in
        range(len(source_dataset_test))]

    return train_loader,test_loader,validation_loader



