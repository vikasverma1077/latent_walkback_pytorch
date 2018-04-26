'''
Created on 21 Nov 2017

@author: vermav1
'''
import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm


import torch
from torchvision import datasets, transforms
from affine_transforms import Rotation, Zoom
from functools import reduce
from operator import __or__
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
#from utils import onehot

from folder import ImageFolder


def get_sampler(labels,n_labels = 10, n=None):
    # Only choose digits in n_labels
    (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

    indices = torch.from_numpy(indices)
    sampler = SubsetRandomSampler(indices)
    return sampler


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode



def get_mnist(location="./", batch_size=64, labels_per_class=100):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    #from utils import onehot

    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    mnist_train = MNIST(location, train=True, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=True,
                                           sampler=get_sampler(mnist_train.train_labels.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=True,
                                             sampler=get_sampler(mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=True,
                                             sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation



def load_data(data_aug,labels_per_class, batch_size,workers,dataset, data_target_dir):
    
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    
    
    else:
        assert False, "Unknow dataset : {}".format(dataset)
    
    if data_aug==1:
        train_transform = transforms.Compose(
                                             [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose(
                                             [ transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        num_classes = 10
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform,  download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform,  download=True)
        
    elif dataset == 'cifar100':
        num_classes = 100
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform,  download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform,  download=True)
        
    elif dataset == 'svhn':
        num_classes = 10
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform,  download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform,  download=True)
        extra_data = datasets.SVHN(data_target_dir, split='extra', transform=train_transform,  download=True)
        
    elif dataset == 'stl10':
        num_classes = 10
        train_data = datasets.STL10(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.STL10(data_target_dir, split='test', transform=test_transform,  download=True)
        
    elif dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
                         num_workers=workers, pin_memory=True, sampler=get_sampler(train_data.labels, num_classes,labels_per_class))
    unlabelled_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                         num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    if dataset=='svhn':
        extra_loader = torch.utils.data.DataLoader(extra_data, batch_size=batch_size, shuffle=True,
                         num_workers=workers, pin_memory=True)
        
        return train_loader, unlabelled_loader, test_loader,  extra_loader, num_classes 
    
    else:
        return train_loader, unlabelled_loader, test_loader,  num_classes 


def load_mnist(data_aug, batch_size, test_batch_size,cuda, data_target_dir):

    if data_aug == 1:
        hw_size = 24
        transform_train = transforms.Compose([
                            transforms.RandomCrop(hw_size),                
                            transforms.ToTensor(),
                            #Rotation(15),                                            
                            #Zoom((0.85, 1.15)),       
                            #transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.CenterCrop(hw_size),                       
                            transforms.ToTensor(),
                            #transforms.Normalize((0.1307,), (0.3081,))
                       ])
    else:
        hw_size = 28
        transform_train = transforms.Compose([
                            transforms.ToTensor(),       
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
    
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}       
    
    
                
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_target_dir, train=True, download=True, transform=transform_train, target_transform=onehot(10)),
        batch_size=batch_size, sampler=get_sampler(mnist_train.labels.numpy(), labels_per_class), **kwargs)
    unlabelled_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_target_dir, train=True, download=True, transform=transform_train, target_transform=onehot(10)),
        batch_size=batch_size, shuffle=True,  **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_target_dir, train=False, transform=transform_test, target_transform=onehot(10)),
        batch_size=test_batch_size, shuffle=True,  **kwargs)
    
    return train_loader, unlabelled_loader,  test_loader


def load_celebA(data_aug, labels_per_class, batch_size, test_batch_size,cuda, data_target_dir):

    train_loader = get_loader(
                data_target_dir, 'train', labels_per_class, batch_size, 64, 2, True)
    
    unlabelled_loader = get_loader(
                data_target_dir, 'train', None, batch_size, 64, 2, True)
    
    test_loader = get_loader(
                data_target_dir, 'test', None, test_batch_size, 64, 2, True)
    
    return train_loader, unlabelled_loader, test_loader    

def get_loader(root, split, labels_per_class, batch_size, scale_size, num_workers=2, shuffle=True):
    
    image_root = os.path.join(root, 'splits', split)

    
    dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.Normalize((0.51, 0.42, 0.37), (0.27, 0.27, 0.27))
        ]))## TODO add target_transform=onehot(n_labels)??
    
    if split == 'train':
        if labels_per_class != None:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers), sampler=get_sampler(dataset.train_labels.numpy(), labels_per_class))
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers)) 
    
    data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader


def main():
    """
    data_loader = get_loader(
                '/u/vermavik/data/CelebA/', 'train', 16, 64, 2, True)
    
    for i, (input, target) in enumerate(data_loader):
        print (input.shape)
        print (target.shape)
        break
    """
    
    train_loader, unlabelled_loader, test_loader, extra_loader, num_classes = load_data(0, labels_per_class=100, batch_size=100,workers=2,dataset='svhn', data_target_dir= '/u/vermavik/data/DARC/svhn')
    #print len(extra_loader), len(train_loader), len(test_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
            print batch_idx
            
    """
    for batch_idx, (data, target) in enumerate(test_loader):
            print data.shape
            break
    for batch_idx, (data, target) in enumerate(extra_loader):
            print data.shape
            break    
    """
    

if __name__ == '__main__':
    main()