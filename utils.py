## copied from semi-supervised wohlert repo##
import torch
from torch import nn
from torch.autograd import Variable
import torch
from torchvision import datasets, transforms

from collections import OrderedDict
import cPickle as pickle
import os
import numpy as np

import matplotlib.pyplot as plt
from operator import __or__
import seaborn as sns
sns.set(color_codes=True)

def plotting(exp_dir):
    # Load the training log dictionary:
    train_dict = pickle.load(open(os.path.join(exp_dir, 'log.pkl'), 'rb'))

    ###########################################################
    ### Make the vanilla train and test loss per epoch plot ###
    ###########################################################
   
    plt.plot(np.asarray(train_dict['train_loss']), label='train_loss')
        
    #plt.ylim(0,2000)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_loss.png' ))
    plt.clf()
    
    
    
    plt.plot(np.asarray(train_dict['test_loss']), label='test_loss')
       
    #plt.ylim(0,100)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'test_loss.png' ))
    plt.clf()
    
        
    plt.plot(np.asarray(train_dict['test_acc']), label='test_acc')
       
    #plt.ylim(0,100)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'test_acc.png' ))
    plt.clf()
   


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())


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


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max


def get_x_z_at_each_step(x, model,temperature, step):
    x = Variable(x.data, requires_grad=False)
    mu, sigma = model.encode(x, step)
    z = model.reparameterize(mu, sigma)
    z_tilde, log_p_reverse, mu, sigma = model.transition( z, temperature, step)
    x_tilde = model.decode(z_tilde,step)
    
    return z, z_tilde, x_tilde, mu


def get_sampler(labels,n_labels, n=None):
    from torch.utils.data.sampler import SubsetRandomSampler
    # Only choose digits in n_labels
    (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

    indices = torch.from_numpy(indices)
    sampler = SubsetRandomSampler(indices)
    return sampler


def load_data_subset(data_aug, batch_size,workers,dataset, data_target_dir, labels_per_class=100):
    ## copied from GibbsNet_pytorch/load.py
    import numpy as np
    from functools import reduce
    from operator import __or__
   
      
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
                         num_workers=workers, pin_memory=True, sampler=get_sampler(train_data.labels,num_classes, labels_per_class))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def get_ssl_results(train_loss, test_loss, test_acc,result_dir, model, num_classes, step, filep, num_epochs, args, labels_per_class, img_shape):
        C = nn.Sequential(
            nn.Linear(args.nl, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax())
        
        
        if torch.cuda.is_available():
            C = C.cuda()
        
        c_optimizer = torch.optim.Adam(C.parameters(), lr=0.001, betas=(0.9,0.99))
        loss_fn = nn.CrossEntropyLoss()
        
        train_loader, test_loader = load_data_subset(data_aug=1, batch_size=32,workers=2,dataset=args.dataset, data_target_dir= '/data/milatmp1/vermavik/data/'+args.dataset+'/', labels_per_class=labels_per_class)
        
        def train(epoch):
            C.train()
            ## train classifier
            train_loss = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                
                x = data
                temperature_forward = args.temperature
                for i in range(step+1):
                    z, z_tilde, x_tilde, mu = get_x_z_at_each_step(x, model,temperature_forward, step=i)
                    x = x.view(-1,img_shape[0], img_shape[1], img_shape[2])#reshape(args.batch_size, n_colors, WIDTH, WIDTH)
                    temperature_forward = temperature_forward * args.temperature_factor;
                
                c_optimizer.zero_grad()
                output = C(mu)
                loss = loss_fn(output, target) 
                loss.backward()
                c_optimizer.step()
                if batch_idx % 100 == 0:
                    str = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0])
                    #print(str)
                    filep.write(str+ '\n')
            
            train_loss += loss.data[0]*target.size(0)
            total += target.size(0)
            
            return train_loss/total
            
            
            

        def test():
        
            C.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)
                x = data
                temperature_forward = args.temperature
                for i in range(step+1):
                    z, z_tilde, x_tilde, mu = get_x_z_at_each_step(x, model,temperature_forward, step=i)
                    x = x.view(-1,img_shape[0], img_shape[1], img_shape[2])#reshape(args.batch_size, n_colors, WIDTH, WIDTH)
                    temperature_forward = temperature_forward * args.temperature_factor;
                
                output = C(mu)
                test_loss += loss_fn(output, target).data[0]*target.shape[0] # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        
            test_loss /= len(test_loader.dataset)
            str = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset))
            #print(str)
            filep.write(str+'\n')
            
            return test_loss,  100. * correct / len(test_loader.dataset)        
        
                 
        for epoch in range(1, num_epochs + 1):
                
            train_l = train(epoch)
            test_l, test_a = test()
            
            train_loss.append(train_l)
            test_loss.append(test_l)
            test_acc.append(test_a)
            
            
            train_log = OrderedDict()
            train_log['train_loss'] = train_loss
            train_log['test_loss']=test_loss
            train_log['test_acc']=test_acc
            
            pickle.dump(train_log, open( os.path.join(result_dir,'log.pkl'), 'wb'))
            plotting(result_dir)
        
        return train_loss, test_loss, test_acc    

def get_ssl_results_vae(train_loss, test_loss, test_acc, result_dir, model, num_classes, filep, num_epochs, args, labels_per_class, img_shape):
        C = nn.Sequential(
            nn.Linear(args.nl, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax())
        
        
        if torch.cuda.is_available():
            C = C.cuda()
        
        c_optimizer = torch.optim.Adam(C.parameters(), lr=0.0001, betas=(0.9,0.99))
        loss_fn = nn.CrossEntropyLoss()
        
        train_loader, test_loader = load_data_subset(data_aug=1, batch_size=32,workers=2,dataset=args.dataset, data_target_dir= '/data/milatmp1/vermavik/data/'+args.dataset+'/', labels_per_class=labels_per_class)
        
        def train(epoch):
            C.train()
            ## train classifier
            train_loss = 0
            total = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                
                mu, sigma = model.encode(data)
                
                c_optimizer.zero_grad()
                output = C(mu)
                loss = loss_fn(output, target) 
                loss.backward()
                c_optimizer.step()
                if batch_idx % 100 == 0:
                    str = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0])
                    #print(str)
                    filep.write(str+ '\n')
                train_loss += loss.data[0]*target.size(0)
                total += target.size(0)
        
            return train_loss/total
            
        def test():
        
            C.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)
                
                mu, sigma = model.encode(data)
                
                output = C(mu)
                test_loss += loss_fn(output, target).data[0]*target.shape[0] # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        
            test_loss /= len(test_loader.dataset)
            str = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset))
            #print(str)
            filep.write(str+'\n')
            
            return test_loss,  100. * correct / len(test_loader.dataset)
        
                
        for epoch in range(1, num_epochs + 1):
                
            train_l = train(epoch)
            test_l, test_a = test()
            
            train_loss.append(train_l)
            test_loss.append(test_l)
            test_acc.append(test_a)
            
            #print (train_loss, test_loss, test_acc)
            
            train_log = OrderedDict()
            train_log['train_loss'] = train_loss
            train_log['test_loss']=test_loss
            train_log['test_acc']=test_acc
            
            pickle.dump(train_log, open( os.path.join(result_dir,'log.pkl'), 'wb'))
            plotting(result_dir)
        
        return train_loss, test_loss, test_acc
