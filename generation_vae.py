'''
Created on Apr 14, 2018

@author: vermavik
'''
import argparse
import numpy as np
import os
import mimir
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm

from viz import *
import sys
#from lib.util import  norm_weight, _p, itemlist,  load_params, create_log_dir, unzip,  save_params
from lib.distributions import log_normal2

from networks_vae import *
from load import *
from utils import *
from distutils.dir_util import copy_tree
from shutil import rmtree
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssl', type=int, default=0, help= 'whether to do ssl on z-space')
    parser.add_argument('--dataset', type=str, default='svhn',
                        help='Name of dataset to use.')
    parser.add_argument('--epochs', type = int, default = 10,
                        help='num of epochs')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size')
    parser.add_argument('--data_aug', type=int, default=0)
    
    
    parser.add_argument('--init_ch', default=128, type=int,
                        help='number of channel in first conv layer')
    parser.add_argument('--nl', type = int, default = 1024,
                        help='Size of Latent Size')
     
    
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Initial learning rate. ' + \
                        'Will be decayed until it\'s 1e-5.')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--optimizer', type = str, default = 'adam',
                        help='optimizer we are going to use!!')
     
                        
    parser.add_argument('--job_id', type=str, default='')
    parser.add_argument('--add_name', type=str, default='')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

##############################
def experiment_name(dataset='celebA',
                    epochs=10,
                    z_size = 256,
                    init_ch =16,
                    job_id=None,
                    add_name=''):
    exp_name = 'vae_'
    exp_name += str(dataset)
    exp_name += '_epochs_'+str(epochs)
    exp_name += '_z_size_'+str(z_size)
    exp_name += '_init_ch_'+str(init_ch)
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)
    
    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name




    
def compute_grad_norm(model):
    norm = 0
    for name, param in model.named_parameters():#base_params:
                if hasattr(param.grad,'data'):
                    g = param.grad.data
                    norm += torch.sum(g.clone()* g.clone())
    return  norm


def compute_param_norm(param):
    norm = torch.sum(param* param)
    return  norm


def reverse_time(scl, shft, sample_drawn, name, shape):
    new_image = np.asarray(sample_drawn).astype('float32').reshape(args.batch_size, shape[0], shape[1], shape[2])
    plot_images(new_image, name)


def compute_loss(x, model, loss_fn):
    
    
    #print 'this'
    mu, sigma = model.encode(x)
    z = model.reparameterize(mu, sigma)
    #print z
    #print ('step', 0)
    #print ('z', np.isnan(z.data.cpu().numpy()).any())
    #print x .requires_grad
    x_tilde = model.decode(z)
    #print x_tilde.shape
    #print x.shape
    #return 1
    x_loss = loss_fn(x_tilde,x.view(-1, x.shape[0]*x.shape[1]*x.shape[2]))## sum over axis=1
    #print x_loss
    #return 1
    
    KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    KLD /= args.batch_size 
    loss = x_loss + 0.001*KLD
    
    return loss, x_loss, KLD



def forward_diffusion(x, model, loss_fn):
    x = Variable(x.data, requires_grad=False)
    mu, sigma = model.encode(x)
    z = model.reparameterize(mu, sigma)
    x_tilde = model.decode(z)
    
    return x_tilde



def train(args, lrate):   
    
    tmp='/Tmp/vermavik/'
    home='/data/milatmp1/vermavik/'

    
    dataset = args.dataset
    #data_source_dir = home+'data/'+dataset+'/'
    """
    if not os.path.exists(data_source_dir):
        os.makedirs(data_source_dir)
    data_target_dir = tmp+'data/CelebA/'
    copy_tree(data_source_dir, data_target_dir)
    """
    ### set up the experiment directories########
    
    
    exp_name=experiment_name(dataset = args.dataset,
                    epochs = args.epochs,
                    z_size = args.nl,
                    init_ch = args.init_ch,
                    job_id=args.job_id,
                    add_name=args.add_name)
    
    #temp_model_dir = tmp+'experiments/HVWB/'+dataset+'/model/'+ exp_name
    #temp_result_dir = tmp+'experiments/HVWB/'+dataset+'/results/'+ exp_name
    model_dir = home+'experiments/HVWB/'+dataset+'/model/'+ exp_name
    result_dir = home+'experiments/HVWB/'+dataset+'/results/'+ exp_name
    
    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    result_path = os.path.join(result_dir , 'out.txt')
    filep = open(result_path, 'w',  buffering=0)
    
    out_str = str(args)
    print(out_str)
    filep.write(out_str + '\n')
    
    
    """   
    #copy_script_to_folder(os.path.abspath(__file__), temp_result_dir)
    result_path = os.path.join(temp_result_dir , 'out.txt')
    filep = open(result_path, 'w')
    
    out_str = str(args)
    print(out_str)
    filep.write(out_str + '\n') 
    """
    sys.setrecursionlimit(10000000)
    args = parse_args()
    print args

   
    ## load the training data
    print args.dataset
    if args.dataset == 'celeba' or args.dataset == 'celebasmall':
        print 'loading celebA'
        train_loader, test_loader=load_celebA(args.data_aug, args.batch_size, args.batch_size, args.cuda, home+'data/'+dataset+'/')
        input_shape = (3,64,64)
    elif args.dataset== 'svhn':
        print 'loading svhn'
        train_loader, test_loader, extra_loader, num_classes = load_data(args.data_aug, args.batch_size ,2, dataset, home+'data/DARC/'+dataset+'/')
        input_shape = (3,32,32)
    elif args.dataset == 'cifar10':
        print 'loading cifar10'
        train_loader, test_loader, num_classes = load_data(args.data_aug, args.batch_size ,2, dataset, home+'data/DARC/'+dataset+'/')
        input_shape = (3,32,32)
    
    for batch_idx, (data, target) in enumerate(train_loader):
    
        Xbatch = data.numpy()
        #print Xbatch
        scl = 1./np.sqrt(np.mean((Xbatch-np.mean(Xbatch))**2))
        shft = -np.mean(Xbatch*scl)
        
        break ### TO DO : calculate statistics on whole data
    
    if dataset == 'svhn' or dataset == 'cifar10' :
        model = VAE(args, imgSize=input_shape)
    else:
        model = VAE_old(args, imgSize=input_shape)
    if args.cuda:
        model.cuda()
    loss_fn = nn.BCELoss()
    """
    if args.optimizer=='sgd':
        optimizer_encoder = optim.SGD(model.encoder_params, lr=args.lr, momentum=args.momentum, weight_decay=0)
        optimizer_transition = optim.SGD(model.transition_params, lr=args.lr, momentum=args.momentum, weight_decay=0)
        optimizer_decoder = optim.SGD(model.decoder_params, lr=args.lr, momentum=args.momentum, weight_decay=0)
    elif args.optimizer=='adam':
        optimizer_encoder = optim.Adam(model.encoder_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer_transition = optim.Adam(model.transition_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer_decoder = optim.Adam(model.decoder_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    """
    if args.optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)
    elif args.optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    
    
    #### for saving metrics for all steps ###
    train_loss = []
    train_x_loss = []
    train_kld = []
    
    #### for saving metrics for each step individually ###
    
    for epoch in range(args.epochs):
        print ('epoch', epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            #print data.min(), data.max()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
           
            #print ('meta_step', meta_step)
            #print encode
            loss, x_loss,  KLD = compute_loss(data, model, loss_fn)
                #meta_cost.append(loss)
            #print compute_param_norm(model.conv_x_z_1.weight.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #total_norm = clip_grad_norm(model.parameters(), args.grad_max_norm)
            #print ('step', meta_step, total_norm)
            #print ('step', meta_step, clip_grad_norm(model.parameters(), 1000000))
            ### store metrics#######
            train_loss.append(loss.data[0])
            train_x_loss.append(x_loss.data[0])
            train_kld.append(KLD.data[0])
            
                
            if np.isnan(loss.data.cpu()[0]) or np.isinf(loss.data.cpu()[0]):
                print loss.data
                print 'NaN detected'
                return 1.
            
            #batch_idx=0
            if args.ssl == 0:
                if batch_idx%500==0:
                    plot_loss_vae(model_dir, train_loss, train_x_loss,  train_kld)
                    
                    data_forward_diffusion = data
                   
                        #print "Forward temperature", temperature_forward
                    data_forward_diffusion = forward_diffusion(data_forward_diffusion, model, loss_fn)
                        #print data_forward_diffusion.shape
                        #data_forward_diffusion = np.asarray(data).astype('float32').reshape(args.batch_size, INPUT_SIZE)
                    data_forward_diffusion = data_forward_diffusion.view(-1, input_shape[0], input_shape[1], input_shape[2])#reshape(args.batch_size, n_colors, WIDTH, WIDTH)
                    plot_images(data_forward_diffusion.data.cpu().numpy(), model_dir + '/' + "batch_" + str(batch_idx) + '_corrupted_' + 'epoch_' + str(epoch) + '_time_step_')
                        
                    
                    print "PLOTTING ORIGINAL IMAGE"
                    temp = data
                    plot_images(temp.data.cpu().numpy() , model_dir + '/' + 'orig_' + 'epoch_' + str(epoch) + '_batch_index_' +  str(batch_idx))
    
                    print "DONE PLOTTING ORIGINAL IMAGE"
                    
                                 
                    #if args.noise == "gaussian":
                    z_sampled = np.random.normal(0.0, 1.0, size=(args.batch_size, args.nl))#.clip(0.0, 1.0)
                    #else:
                    #    z_sampled = np.random.binomial(1, 0.5, size=(args.batch_size, args.nl))
    
                    
                    z = torch.from_numpy(np.asarray(z_sampled).astype('float32'))
                    if args.cuda:
                        z = z.cuda()
                        z = Variable(z)
                    x_sampled  = model.decode(z)
                        #print 'On step number, using temperature', i, temperature
                    reverse_time(scl, shft, x_sampled.data.cpu().numpy(), model_dir + '/batch_index_' + str(batch_idx) + '_inference_' + 'epoch_' + str(epoch), input_shape)
        
        if args.ssl==1:    
            get_ssl_results_vae(result_dir, model, num_classes, train_loader, test_loader, filep = filep, num_epochs=100, args=args, num_of_batches= 40, img_shape= input_shape)
    filep.close()

                    
                
if __name__ == '__main__':
    args= parse_args()
    train(args, lrate=0.0001)
    pass

