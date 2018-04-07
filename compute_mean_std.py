'''
Created on Mar 27, 2018

@author: vermavik
'''
#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import os
import sys
import math

import numpy as np


from folder import ImageFolder

"""
data = dset.CIFAR10(root='/u/vermavik/data/DARC/cifar10', train=True, download=True,
                    transform=transforms.ToTensor()).train_data
data = data.astype(np.float32)/255.
print data.shape
means = []
stdevs = []
for i in range(3):
    pixels = data[:,:,:,i].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
"""


def load_celebA(data_aug, batch_size, test_batch_size,cuda, data_target_dir):

    train_loader = get_loader(
                data_target_dir, 'train', batch_size, 64, 2, True)
    
    test_loader = get_loader(
                data_target_dir, 'test', test_batch_size, 64, 2, True)
    
    return train_loader, test_loader    

def get_loader(root, split, batch_size, scale_size, num_workers=2, shuffle=True):
    
    image_root = os.path.join(root, 'splits', split)

    dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    data_loader.shape = [int(num) for num in dataset[0][0].size()]
    print data_loader.shape
    return data_loader




data_loader = get_loader(
            '/u/vermavik/data/CelebA/', 'train', 1000, 64, 2, True)


data_all= np.zeros((162770,3,64,64))#162770

start_idx = 0
end_idx = 0
for batch_idx, (data, target) in enumerate(data_loader):
        print batch_idx
    
        data_size = data.numpy().shape[0]
        #print data_size
        data_all[start_idx:start_idx+data_size,:,:,:]= data.numpy()
        start_idx += data_size
        print start_idx
    
print data_all.shape

means = []
stdevs = []
for i in range(3):
    pixels = data_all[:,i,:,:].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
    
print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
#transforms.Normalize(mean = [0.510746429411343, 0.41528771647995555, 0.36520285671498554], std = [0.2969496579891956, 0.2695166542024482, 0.26523371031302384])
