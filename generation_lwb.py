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

from networks_lwb import *
from load import *
from distutils.dir_util import copy_tree
from shutil import rmtree
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='celebasmall',
                        help='Name of dataset to use.')
    parser.add_argument('--activation', type=str, default='leakyrelu',
                        help='activation function to use in the network except the last layer of decoder')
    parser.add_argument('--epochs', type = int, default = 10,
                        help='num of epochs')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size')
    parser.add_argument('--data_aug', type=int, default=0)
    
    
    parser.add_argument('--init_ch', default=32, type=int,
                        help='number of channel in first conv layer')
    parser.add_argument('--enc_fc_size', default=128, type=int,
                        help='size of fc layer before mu, sigma')
    parser.add_argument('--nl', type = int, default = 128,
                        help='Size of Latent Size')
    parser.add_argument('--transition_size', default=256, type=int,
                        help='size of transition layers')
    parser.add_argument('--transition_steps', default=5, type=int,
                        help='number of transition steps')
    
    parser.add_argument('--use_decoder', type = bool, default = True,
                        help='whether should we use decoder')
    parser.add_argument('--use_encoder', type = bool, default = True,
                        help='whether should we use encoder')
   
    parser.add_argument('--encode_every_step', type=int, default=0)
    parser.add_argument('--alpha1', type=float, default=1.0,help='coefficient for reconstruction loss')
    parser.add_argument('--alpha2', type=float, default= 1.0,help='coefficient for log_p_reverse')
    parser.add_argument('--alpha3', type=float, default=1.0,help='coefficient for KLD')
    
    
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Initial learning rate. ' + \
                        'Will be decayed until it\'s 1e-5.')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--optimizer', type = str, default = 'sgd',
                        help='optimizer we are going to use!!')
    parser.add_argument('--grad_max_norm', type=float, default=5.0,
                        help='max value of grad norm used for gradient clipping')
    
    parser.add_argument('--noise_prob', default=0.1, type=float,
                        help='probability for bernouli distribution of adding noise of 1 to each input')
    parser.add_argument('--avg', default=0, type=float)
    parser.add_argument('--std', default=1., type=float)
    parser.add_argument('--noise', default='gaussian', choices=['gaussian', 'binomial'])
    
    
    parser.add_argument('--num_steps', type=int, default=1,
                        help='Number of transition steps.')
    parser.add_argument('--extra_steps', type = int, default = 10,
                        help='Number of extra steps to sample at temperature 1')
    parser.add_argument('--meta_steps', type = int, default = 10,
                        help='Number of times encoder-decoder is repeated')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Standard deviation of the diffusion process.')
    parser.add_argument('--temperature_factor', type = float, default = 1.1,
                        help='How much temperature must be scaled')
    parser.add_argument('--sigma', type = float, default = 1.0,
                        help='How much Noise should be added at step 1')
   
    
                        
    parser.add_argument('--job_id', type=str, default='')
    parser.add_argument('--add_name', type=str, default='')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

##############################
def experiment_name(dataset='celebA',
                    act = 'relu',
                    meta_steps=10,
                    sigma = 0.0001,
                    temperature_factor = 1.1,
                    alpha1 = 1.0,
                    alpha2 = 1.0,
                    alpha3 = 1.0,
                    grad_norm_max = 5.0,
                    epochs=10,
                    z_size = 256,
                    init_ch =16,
                    enc_fc_size =128,
                    transition_size = 256,
                    transition_steps =3, 
                    job_id=None,
                    add_name=''):
    exp_name = str(dataset)
    exp_name += '_act_'+str(act)
    exp_name += '_meta_steps_'+str(meta_steps)
    exp_name += '_sigma_'+str(sigma)
    exp_name += '_temperature_factor_'+str(temperature_factor)
    exp_name += '_alpha1_'+str(alpha1)
    exp_name += '_alpha2_'+str(alpha2)
    exp_name += '_alpha3_'+str(alpha3)
    exp_name += '_grad_norm_max_'+str(grad_norm_max)
    exp_name += '_epochs_'+str(epochs)
    exp_name += '_z_size_'+str(z_size)
    exp_name += '_init_ch_'+str(init_ch)
    exp_name += '_enc_fc_size_'+str(enc_fc_size)
    exp_name += '_transition_size_'+str(transition_size)
    exp_name += '_transition_steps_'+str(transition_steps)
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


def compute_loss(x, z, model, loss_fn, start_temperature, meta_step, encode= False):
    
    
    temperature = start_temperature
    if encode==True:
        #print 'this'
        mu, sigma = model.encode(x, meta_step)
        z = model.reparameterize(mu, sigma)
    #print z
    #print ('step', 0)
    #print ('z', np.isnan(z.data.cpu().numpy()).any())
    #print x .requires_grad
    z_tilde, log_p_reverse,mu, sigma = model.transition( z, temperature, meta_step)
    x_tilde = model.decode(z_tilde, meta_step)
    #print x_tilde.shape
    #print x.shape
    #return 1
    x_loss = loss_fn(x_tilde,x.view(-1, x.shape[0]*x.shape[1]*x.shape[2]))## sum over axis=1
    #print x_loss
    #return 1
    loss =  -log_p_reverse*args.alpha2 + x_loss*args.alpha1 
    
    if meta_step==args.meta_steps-1:
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        KLD /= args.batch_size 
        loss = loss + KLD*args.alpha3
    else:
        KLD = None
    #print ('x_loss', x_loss)
    #print ('log_p_reverse', log_p_reverse)
    x_states = [x_tilde]
    z_states = [z_tilde]
    
    return loss, x_loss, log_p_reverse, KLD,  z, z_tilde, x_tilde



def forward_diffusion(x, model, loss_fn,temperature, step):
    x = Variable(x.data, requires_grad=False)
    mu, sigma = model.encode(x, step)
    z = model.reparameterize(mu, sigma)
    z_tilde, log_p_reverse, sigma, h2 = model.transition( z, temperature, step)
    x_tilde = model.decode(z_tilde,step)
    x_loss = loss_fn (x_tilde,x)## sum over axis=1
    total_loss = log_p_reverse + x_loss
    
    return x_tilde, total_loss, z_tilde, x_loss, log_p_reverse, sigma, h2



def train(args, lrate):   
    
    tmp='/Tmp/vermavik/'
    home='/u/vermavik/'
    
    
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
                    act = args.activation,
                    meta_steps = args.meta_steps,
                    sigma = args.sigma,
                    temperature_factor = args.temperature_factor,
                    alpha1 = args.alpha1,
                    alpha2 = args.alpha2,
                    alpha3 = args.alpha3,
                    grad_norm_max = args.grad_max_norm,
                    epochs = args.epochs,
                    z_size = args.nl,
                    init_ch = args.init_ch,
                    enc_fc_size = args.enc_fc_size,
                    transition_size = args.transition_size,
                    transition_steps = args.transition_steps,
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
    
    if args.dataset == 'cifar10':
        model = Net_cifar(args, input_shape=input_shape)
    elif args.dataset == 'svhn':
        print ('this')
        model = Net_svhn(args, input_shape=input_shape)
    else:
        model = Net(args, input_shape=input_shape)
    if args.cuda:
        model.cuda()
    loss_fn = nn.BCELoss()
    if args.optimizer=='sgd':
        optimizer_encoder = optim.SGD(model.encoder_params, lr=args.lr, momentum=args.momentum, weight_decay=0)
        optimizer_transition = optim.SGD(model.transition_params, lr=args.lr, momentum=args.momentum, weight_decay=0)
        optimizer_decoder = optim.SGD(model.decoder_params, lr=args.lr, momentum=args.momentum, weight_decay=0)
    elif args.optimizer=='adam':
        optimizer_encoder = optim.Adam(model.encoder_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer_transition = optim.Adam(model.transition_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer_decoder = optim.Adam(model.decoder_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
   
   
    
    #### for saving metrics for all steps ###
    train_loss = []
    train_x_loss = []
    train_log_p_reverse = []
    train_kld = []
    
    #### for saving metrics for each step individually ###
    train_loss_each_step = [[]]
    train_x_loss_each_step = [[]]
    train_log_p_reverse_each_step = [[]]
    #train_kld_each_step = [[]]
    for i in range(args.meta_steps-1):
        train_loss_each_step.append([])
        train_x_loss_each_step.append([])
        train_log_p_reverse_each_step.append([])
        #train_kld_each_step.append([])
    
    for epoch in range(args.epochs):
        print ('epoch', epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            temperature_forward = args.temperature
            x = data
            z = None
            encode = True
            for meta_step in range(0, args.meta_steps):
                #print ('meta_step', meta_step)
                #print encode
                loss, x_loss, log_p_reverse, KLD, z, z_tilde, x_tilde = compute_loss(x, z , model, loss_fn, temperature_forward, meta_step, encode=encode)
                    #meta_cost.append(loss)
                #print compute_param_norm(model.conv_x_z_1.weight.data)
                optimizer_encoder.zero_grad()
                optimizer_transition.zero_grad()
                optimizer_decoder.zero_grad()
                loss.backward()
                #total_norm = clip_grad_norm(model.parameters(), args.grad_max_norm)
                #print ('step', meta_step, total_norm)
                if encode==True:
                    optimizer_encoder.step()
                optimizer_transition.step()
                optimizer_decoder.step()
                
                #print ('step', meta_step, clip_grad_norm(model.parameters(), 1000000))
                ### store metrics#######
                train_loss.append(loss.data[0])
                train_x_loss.append(x_loss.data[0])
                train_log_p_reverse.append(-log_p_reverse.data[0])
                if KLD is not None:
                    train_kld.append(KLD.data[0])
                
                
                #### store metrices for each step separately###
                train_loss_each_step[meta_step].append(loss.data[0])
                train_x_loss_each_step[meta_step].append(x_loss.data[0])
                train_log_p_reverse_each_step[meta_step].append(-log_p_reverse.data[0])
                #if KLD is not None:
                #    train_kld_each_step[meta_step].append(KLD.data[0])
                    
                if args.meta_steps>1:
                    x =  Variable(x_tilde.data.view(-1, input_shape[0], input_shape[1], input_shape[2]), requires_grad=False)
                    z = Variable(z_tilde.data, requires_grad=False)
                    if args.encode_every_step==0:
                        encode = False
                    temperature_forward *= args.temperature_factor
                
                
            if np.isnan(loss.data.cpu()[0]) or np.isinf(loss.data.cpu()[0]):
                print loss.data
                print 'NaN detected'
                return 1.
            
            #batch_idx=0
            if batch_idx%500==0:
                plot_loss(model_dir, train_loss, train_x_loss, train_log_p_reverse, train_kld, train_loss_each_step, train_x_loss_each_step, train_log_p_reverse_each_step, args.meta_steps)
                temperature = args.temperature * (args.temperature_factor ** (args.num_steps*args.meta_steps -1 ))
                temperature_forward = args.temperature
                #print 'this'
                
                
                data_forward_diffusion = data
                for num_step in range(args.num_steps * args.meta_steps):
                    #print "Forward temperature", temperature_forward
                    data_forward_diffusion, _, _, _, _, _, _ = forward_diffusion(data_forward_diffusion, model, loss_fn,temperature_forward, num_step)
                    #print data_forward_diffusion.shape
                    #data_forward_diffusion = np.asarray(data).astype('float32').reshape(args.batch_size, INPUT_SIZE)
                    data_forward_diffusion = data_forward_diffusion.view(-1, input_shape[0], input_shape[1], input_shape[2])#reshape(args.batch_size, n_colors, WIDTH, WIDTH)
                    if num_step%2==0:
                        plot_images(data_forward_diffusion.data.cpu().numpy(), model_dir + '/' + "batch_" + str(batch_idx) + '_corrupted_' + 'epoch_' + str(epoch) + '_time_step_' + str(num_step))
                    
                    temperature_forward = temperature_forward * args.temperature_factor;
                
                print "PLOTTING ORIGINAL IMAGE"
                temp = data
                plot_images(temp.data.cpu().numpy() , model_dir + '/' + 'orig_' + 'epoch_' + str(epoch) + '_batch_index_' +  str(batch_idx))

                print "DONE PLOTTING ORIGINAL IMAGE"
                
                             
                if args.noise == "gaussian":
                    z_sampled = np.random.normal(0.0, 1.0, size=(args.batch_size, args.nl))#.clip(0.0, 1.0)
                else:
                    z_sampled = np.random.binomial(1, 0.5, size=(args.batch_size, args.nl))

                temperature = args.temperature * (args.temperature_factor ** (args.num_steps*args.meta_steps - 1))

                z = torch.from_numpy(np.asarray(z_sampled).astype('float32'))
                if args.cuda:
                    z = z.cuda()
                    z = Variable(z)
                for i in range(args.num_steps*args.meta_steps):# + args.extra_steps):
                    z_new_to_x, z_to_x, z_new  = model.sample(z, temperature, args.num_steps*args.meta_steps -i - 1)
                    #print 'On step number, using temperature', i, temperature
                    if i%2==0:
                        reverse_time(scl, shft, z_new_to_x.data.cpu().numpy(), model_dir + '/batch_index_' + str(batch_idx) + '_inference_' + 'epoch_' + str(epoch) + '_step_' + str(i), input_shape)
                    
                    if temperature == args.temperature:
                        temperature = temperature
                    else:
                        temperature /= args.temperature_factor
                    z = z_new
                
 
if __name__ == '__main__':
    args= parse_args()
    train(args, lrate=0.0001)
    pass

