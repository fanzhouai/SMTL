"""This module defines the function and classes to control datasets

date-of-creation: 2019-02-16
"""

import os
import zipfile
import shutil
import urllib
import tarfile

import numpy as np
from PIL import Image
from skimage.transform import resize as skresize
from skimage.io import imread as skimread
from skimage.color import rgb2gray as skrgb2gray

from gensim.corpora import Dictionary as GensimDict

import torch
from torch.utils.data import Dataset

import torchvision


"""
IMAGE DATASETS
"""


class Local_Dataset_digit(Dataset):
    """TODO: docstring
    """

    def __init__(self, data_name, set, data_path, transform, num_samples=None):
        super(Local_Dataset_digit, self).__init__()
        self.data_path = data_path
        self.data_name = data_name
        self.set = set
        self.transform = transform
        self.num_samples = num_samples


        if self.data_name == 'usps':

            self.inputs , self.targets =  self._USPS()


        elif self.data_name== 'm_mnist' or self.data_name == 'mnist':
            if self.set =='train' or self.set=='validation':
                imgs, lbls = torch.load(open('data/mnist/processed/training.pt','rb'))

            elif self.set == 'test':
                imgs, lbls = torch.load(open('data/mnist/processed/test.pt','rb'))
                print (imgs.size())



            if self.data_name =='mnist':
                self.inputs , self.targets = imgs , lbls
            else:
                self.inputs , self.targets = self._create_m_mnist( imgs, lbls)

        self.inputs, self.targets = self._select_data()

        print(self.data_name+' size '+str(self.inputs.size())+str( self.targets.size())+str([torch.min(self.inputs), torch.max(self.inputs)]))


    def __getitem__(self, index):

        img = self.inputs[index]
        lbl = self.targets[index]
        # img from tensor converted to numpy for applying a transformation:
        img = Image.fromarray(img.numpy())
        # img = self._checklist(img)

        if self.transform is not None:
            # convert back to tensor will be done as transform
            img = self.transform(img)

        return img, lbl

    def __len__(self):

        return len(self.inputs)

    # def _checklist(self, img):
    #     if torch.max(img)>=255.0: self.inputs/=255.0
    #     # img =()
    #     if img.ndim==4 and img.shape[1] ==1: img = np.squeeze(img, axis=1)
    #     return  img

    def _select_data (self):
        try :
            inputs , targets = self.inputs.numpy(), self.targets.numpy()
        except AttributeError:
            inputs, targets = self.inputs, self.targets
            pass
        if len(
            inputs) < self.num_samples:
            print ("! requested number of samples {:d} exceed the available data {:d}!! The maximum number to request is {:d}".format(self.num_samples, len(inputs), len(inputs)))
            self.num_samples = len(inputs)
        s_inputs, s_targets = [],[]
        for i in range(10):
            indx = np.where((targets).astype('uint8')==i)[0]

            if self.set == 'validation':
                s_inputs.append(inputs[indx][-100:])
                s_targets.append(targets[indx][-100:])
            elif self.set=='train' or self.set=='test':

                s_inputs.append(inputs[indx][:int(self.num_samples/10)])
                s_targets.append(targets[indx][:int(self.num_samples/10)])

        s_inputs = np.concatenate(s_inputs, axis=0)
        s_targets = np.concatenate(s_targets, axis=0)
        s_inputs = torch.tensor(s_inputs)
        s_targets = torch.tensor(s_targets, dtype=torch.long)
        return s_inputs,s_targets

    def _create_m_mnist(self,imgs,lbls):
        imgs, lbls = imgs.numpy(), lbls.numpy()
        print (imgs.shape)
        assert  len(imgs) == len(lbls)

        def _compose_image(digit, background):
            """Difference-blend a digit and a random patch from a background image."""

            w, h, _ = background.shape
            dw, dh, _ = digit.shape
            x = np.random.randint(0, w - dw)
            y = np.random.randint(0, h - dh)

            bg = background[x:x + dw, y:y + dh]
            return np.abs(bg - digit).astype(np.uint8)

        def _mnist_to_img(x):
            """Binarize MNIST digit and convert to RGB."""
            x = (x > 0).astype(np.float32)
            d = x.reshape([28, 28, 1]) * 255
            return np.concatenate([d, d, d], 2)

        def _create_mnistm(X, background_data):
            """
            Give an array of MNIST digits, blend random background patches to
            build the MNIST-M dataset as described in
            http://jmlr.org/papers/volume17/15-239/15-239.pdf
            """
            rand = np.random.RandomState(42)
            X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
            for i in range(X.shape[0]):

                bg_img = rand.choice(background_data)
                while bg_img is None: bg_img = rand.choice(background_data)
                d = _mnist_to_img(X[i])
                d = _compose_image(d, bg_img)
                X_[i] = d

            return X_

        # # import pdb ;pdb.set_trace()
        # if  not os.path.isfile(self.data_path+'/mnist_m_data.pt'):

        BST_PATH = self.data_path+'/BSR_bsds500.tgz'

        if 'BSR_bsds500.tgz' not in os.listdir(self.data_path):
            urllib.request.urlretrieve('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz',self.data_path+'/BSR_bsds500.tgz')

        f = tarfile.open(BST_PATH)
        train_files = []
        for name in f.getnames():
            if self.set=='train' or self.set=='validation':
                the_set = 'train'
            else: the_set='test'
            if name.startswith('BSR/BSDS500/data/images/'+the_set+'/'):
                    train_files.append(name)


        background_data = []
        for name in train_files:
            try:
                fp = f.extractfile(name)
                bg_img = Image.open(fp)
                background_data.append(np.array(bg_img))
            except:
                continue

        # os.remove(self.data_path+'/BSR_bsds500.tgz')

        train = _create_mnistm(imgs, background_data)
        train = np.mean(train, axis=3)
        train = train.reshape(-1, 28, 28)
        train = train/255.0

        # if self.set=='train':train, lbls = self._select_data(train, lbls)
        train = torch.tensor(train)
        lbls = torch.tensor(lbls, dtype=torch.long)


        #     with open(self.data_path+'/mnist_m_data'+self.set+'.pt', 'wb') as f:
        #         torch.save((train, lbls), f)
        # else:
        #     with open(self.data_path+'/mnist_m_data'+self.set+'.pt', 'rb') as f:
        #         train, lbls = torch.load(f)


        return train, lbls


    def _USPS (self):
        def resize_and_scale(img, size, scale):
            img = skresize(img, size)
            return 1 - (np.array(img, "float32") / scale)

        # if  os.path.isfile(self.data_path+'/USPS'+set+'.pt'):
        sz = (28, 28)
        imgs_usps = []
        lbl_usps = []
        if 'USPdata.zip' not in os.listdir(self.data_path): urllib.request.urlretrieve('https://github.com/darshanbagul/USPS_Digit_Classification/raw/master/USPSdata/USPSdata.zip', self.data_path+'/USPSdata.zip')
        zip_ref = zipfile.ZipFile(self.data_path + '/USPSdata.zip', 'r')
        zip_ref.extractall(self.data_path)
        zip_ref.close()
        if self.set =='train' or self.set=='validation':
            for i in range(10):
                label_data = self.data_path+'/Numerals/'+ str(i) + '/'
                img_list = os.listdir(label_data)
                for name in img_list:
                    if '.png' in name:
                        img = skimread(label_data + name)
                        img = skrgb2gray(img)
                        resized_img = resize_and_scale(img, sz, 255)
                        imgs_usps.append(resized_img.flatten())
                        lbl_usps.append(i)

        elif self.set =='test':
            test_path = self.data_path+'/Test/'
            strt = 1
            for lbl, cntr in enumerate(range(151,1651, 150)):

                    for i in range(strt, cntr):
                        i = format(i, '04d')
                        img = skimread(os.path.join(test_path, 'test_'+str(i)+'.png'))
                        img = skrgb2gray(img)
                        resized_img = resize_and_scale(img, sz, 255)
                        imgs_usps.append(resized_img.flatten())
                        lbl_usps.append(9-lbl)
                    strt= cntr

        # os.remove(self.data_path+'/USPSdata.zip')
        shutil.rmtree(self.data_path+'/Numerals')
        shutil.rmtree(self.data_path + '/Test')
        imgs_usps, lbl_usps = np.asarray(imgs_usps).reshape(-1,28,28), np.asarray(lbl_usps)
        lbl_usps = torch.tensor(lbl_usps,dtype= torch.long)
        imgs_usps = torch.tensor(imgs_usps)

        # torch.save((imgs_usps,lbl_usps), open(self.data_path+'/USPS'+set+'.pt','wb'))
        #
        # else:
        #     imgs_usps, lbl_usps = torch.load(open(self.data_path+'/USPS'+set+'.pt','rb'))

        return imgs_usps,lbl_usps


class Local_SVHN(torchvision.datasets.SVHN):
    """TODO: docstring
    """

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False, num_smpl=None):
        super(Local_SVHN, self).__init__(
            root, split, transform, target_transform, download)
        self.data = self.data[:num_smpl]
        self.labels = self.labels[:num_smpl]
        self.maximum = np.max(self.data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        # Convert to grayscale
        img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


"""
TEXT DATASETS
"""


def parse_processed_amazon_dataset(task_files, max_words=10000):
    """
    Code inspired by:
    https://github.com/sclincha/xrce_msda_da_regularization
    """
    datasets = {}
    dico = GensimDict()
    print("Parsing", task_files)

    # First pass on document to build dictionary
    for fname in task_files:
        with open(fname, 'r') as f:
            for l in f:
                tokens = l.split(' ')
                tokens_list = []
                for tok in tokens[:-1]:
                    ts, tfreq = tok.split(':')
                    freq = int(tfreq)
                    tokens_list += [ts] * freq
                dico.doc2bow(tokens_list, allow_update=True)

    # Preprocessing_options
    dico.filter_extremes(no_below=2, keep_n=max_words)
    dico.compactify()

    for fname in task_files:
        X, Y = [], []

        with open(fname, 'r') as f:
            for docid, l in enumerate(f):
                tokens = l.split(' ')
                label_string = tokens[-1]
                tokens_list = []
                for tok in tokens[:-1]:
                    ts, tfreq = tok.split(':')
                    freq = int(tfreq)
                    tokens_list += [ts] * freq
                count_list = dico.doc2bow(tokens_list, allow_update=False)

                idx, freqs = list(zip(*count_list))
                one_hot = np.zeros(max_words)
                one_hot[list(idx)] = np.array(freqs)

                X.append((docid, one_hot))

                #Preprocess Label
                ls, lvalue = label_string.split(':')
                if ls == "#label#":
                    if lvalue.rstrip() == 'positive':
                        Y.append(1)
                    elif lvalue.rstrip() == 'negative':
                        Y.append(0)
                    else:
                        raise Exception("Invalid Label Value")
                else:
                    raise Exception('Invalid Format')

        datasets[os.path.split(os.path.split(fname)[0])[-1]] = (X, Y)

    return datasets, dico


class TextDataset(Dataset):
    """Defines a small PyTorch proxy for amazon review dataset

    Args:
        x (list of tuple): tuple of (docid, features)
        y (array of int): array of labels (positives/negatives)
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]
        x, y = data[0][-1], torch.FloatTensor([data[1]])
        x = torch.from_numpy(x).float()
        return (x, y)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import itertools

    root = 'data/processed_acl/'
    tasks = ['books', 'dvd', 'electronics', 'kitchen']
    in_file = 'labelled.review'

    files = [os.path.join(root, os.path.join(task, in_file)) for task in tasks]

    datasets, dico = parse_processed_amazon_dataset(files)

    books = TextDataset(list(zip(datasets['books'])))
    print(books[0])

















