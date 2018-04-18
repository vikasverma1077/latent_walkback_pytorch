'''
Created on Apr 15, 2018

@author: vermavik
'''
## code merged from train_dsprite_diff_bnorm.py and disentanglement_dsprite_vae_kim_conv.py

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
import tensordataset


from viz import *
import sys
#from lib.util import  norm_weight, _p, itemlist,  load_params, create_log_dir, unzip,  save_params
from lib.distributions import log_normal2

from networks_disentanglement import *
from load import *
from distutils.dir_util import copy_tree
from shutil import rmtree
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 10,
                        help='num of epochs')
    parser.add_argument('--train_size', default=700000, type=int,
                        help='subset size from dsprite used for training')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size')
    parser.add_argument('--data_aug', type=int, default=0)
    
    parser.add_argument('--nl', type = int, default = 10,
                        help='Size of Latent Size')
    parser.add_argument('--transition_size', default= 10, type=int,
                        help='size of transition layers')
    parser.add_argument('--transition_steps', default=5, type=int,
                        help='number of transition steps')
    
    parser.add_argument('--use_decoder', type = bool, default = True,
                        help='whether should we use decoder')
    parser.add_argument('--use_encoder', type = bool, default = True,
                        help='whether should we use encoder')
   
    parser.add_argument('--encode_every_step', type=int, default=1)
    parser.add_argument('--alpha1', type=float, default=1.0,help='coefficient for reconstruction loss')
    parser.add_argument('--alpha2', type=float, default= 1.0,help='coefficient for log_p_reverse')
    parser.add_argument('--alpha3', type=float, default=1.0,help='coefficient for KLD')
    
    
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Initial learning rate. ' + \
                        'Will be decayed until it\'s 1e-5.')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--optimizer', type = str, default = 'adam',
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
def experiment_name_lwb( train_size=10000,
                    meta_steps=10,
                    sigma = 1.0,
                    temperature_factor = 1.1,
                    alpha1 = 1.0,
                    alpha2 = 1.0,
                    alpha3 = 1.0,
                    grad_max_norm = 5.0,
                    epochs=10,
                    z_size = 10,
                    transition_size = 10,
                    transition_steps =5, 
                    job_id=None,
                    add_name=''):
    exp_name = str('lwb')
    exp_name += '_train_size_'+str(train_size)
    exp_name += '_meta_steps_'+str(meta_steps)
    exp_name += '_sigma_'+str(sigma)
    exp_name += '_temperature_factor_'+str(temperature_factor)
    exp_name += '_alpha1_'+str(alpha1)
    exp_name += '_alpha2_'+str(alpha2)
    exp_name += '_alpha3_'+str(alpha3)
    exp_name += 'grad_max_norm'+str(grad_max_norm)
    exp_name += '_epochs_'+str(epochs)
    exp_name += '_z_size_'+str(z_size)
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


dataset_zip = np.load('/u/vermavik/github/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

print('Keys in the dataset:', dataset_zip.keys())
imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']

metadata = dataset_zip['metadata'][()]

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata['latents_sizes']
print (latents_sizes)
latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

def latent_to_index(latents):
  return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):
  samples = np.zeros((size, latents_sizes.size))
  for lat_i, lat_size in enumerate(latents_sizes):
    samples[:, lat_i] = np.random.randint(lat_size, size=size)

  return samples

args = parse_args()

# Sample latents randomly
latents_train = sample_latent(size=args.train_size)
latents_test = sample_latent(size=1000)

# Select images
imgs_train = imgs[latent_to_index(latents_train)]
imgs_test = imgs[latent_to_index(latents_test)]

train_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_train).type(torch.FloatTensor))
test_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_test).type(torch.FloatTensor))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


def compute_loss(x, z, model, loss_fn, start_temperature, meta_step, encode= False, imgSize=1*64*64):
    
    
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
    x_loss = loss_fn(x_tilde,x.view(-1, 1*64*64))## sum over axis=1
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

def get_x_z_at_each_step(x, model,temperature, step):
    x = Variable(x.data, requires_grad=False)
    mu, sigma = model.encode(x, step)
    z = model.reparameterize(mu, sigma)
    z_tilde, log_p_reverse, sigma, h2 = model.transition( z, temperature, step)
    x_tilde = model.decode(z_tilde,step)
    
    return x_tilde, z, mu



def get_disentanglement_score(epoch, model_dir, filep, step):
    ### disentanglement computation#####
    ## step : step in the forward computation, at which z is extracted
    print ('step=', step)
    z_size = args.nl
    model = torch.load(model_dir+'/model.pt')
    
    
    def select_fixed_latent_factor_index(latents_sizes):
        fixed_latent_factor_index = np.random.randint(low=1, high= latents_sizes.shape[0])# low=1 becaues first index is color which has only one value
        return fixed_latent_factor_index
    
    
    
    
    def get_std_over_large_subset(model, step, subset_size=100000):
        latents_sampled = sample_latent(size=subset_size)
        #print (latents_sampled.shape)
        indices_sampled = latent_to_index(latents_sampled)
        img_sampled = imgs[indices_sampled]
    
        img_sampled = Variable(torch.from_numpy(img_sampled).type(torch.FloatTensor).cuda(), volatile=True)
        
        x = img_sampled.unsqueeze(1)
        temperature_forward = args.temperature
        for i in range(step):
            x, z, mu = get_x_z_at_each_step(x, model,temperature_forward, step=i)
            x = x.view(-1,1, 64, 64)#reshape(args.batch_size, n_colors, WIDTH, WIDTH)
            temperature_forward = temperature_forward * args.temperature_factor;
        #print (mu.shape)
        
        std_each_z = mu.data.std(dim=0)
        return std_each_z
    
    std_each_z = get_std_over_large_subset(model,step, subset_size=1000)
    #print (std_each_z)
        
    def get_x(model, fixed_latent_factor_index, batch_size, std_each_z, step):
        latents_sampled = sample_latent(size=batch_size)
        
        #for i in range(batch_size):
        fixed_latent_factor_class = np.random.randint(low=0, high= latents_sizes[fixed_latent_factor_index])
        latents_sampled[:, fixed_latent_factor_index] = fixed_latent_factor_class
            
        
        indices_sampled = latent_to_index(latents_sampled)
        img_sampled = imgs[indices_sampled]
        
        
        img_sampled_a = Variable(torch.from_numpy(img_sampled).type(torch.FloatTensor).cuda(), volatile=True)
        x = img_sampled_a.unsqueeze(1)
        temperature_forward = args.temperature
        for i in range(step):
            x, z, mu = get_x_z_at_each_step(x, model,temperature_forward, step=i)
            x = x.view(-1,1, 64, 64)#reshape(args.batch_size, n_colors, WIDTH, WIDTH)
            temperature_forward = temperature_forward * args.temperature_factor;
    
        mu = mu.data
        
        normalized_z = mu/std_each_z
        
        variance_z = normalized_z.var(dim=0)
        
        variance_z = variance_z.cpu()
        x = np.argmin(variance_z)
        
        return x
    
    
    ########################################
    ### create train data for classifier####
    ########################################
    check = False ## check to ensure that train_x has all possible indexes of Z's and train_y has all possible indexes of Factors (5 in case of dsprite)
    j = 0
    while (check== False):
        j +=1
        num_samples = 10000
        x = np.zeros(num_samples)
        y = np.zeros(num_samples)
        
        for i in range(num_samples):
            index = select_fixed_latent_factor_index(latents_sizes) ## select the index of the latent factor that has to be kept fixed for the batch of size L in paper
            y[i]= index
            x[i] = get_x(model, index, 100, std_each_z,step)##  difference defined at point 3 page 7 https://openreview.net/pdf?id=Sy2fzU9gl
            
        train_x = x
        train_y = y
        
        #print (train_x)
        #print (train_y)
        unique_x = np.unique(train_x)
        unique_y = np.unique(train_y)
        
        if unique_x.size == args.nl and unique_y.size == 5:
            check = True
            print ('number of times sampling done %{}'.format(j))
        
    print (train_x.shape)
    print (train_y.shape)
    
    
    """
    ### create test data for classifier###
    ######################################
    
    check = False ## check to ensure that train_x has all possible indexes of Z's and train_y has all possible indexes of Factors (5 in case of dsprite)
    i =0 
    while (check== False):
    
        num_samples = 10000
        x = np.zeros(num_samples)
        y = np.zeros(num_samples)## target labels for the disentanglement classifier
        
        for i in range(num_samples):
            index = select_fixed_latent_factor_index(latents_sizes) ## select the index of the latent factor that has to be kept fixed for the batch of size L in paper
            y[i]= index
            x[i] = get_x(model, index, 100, std_each_z, step)##  difference defined at point 3 page 7 https://openreview.net/pdf?id=Sy2fzU9gl
        #print (y)
        #print (x)  
        test_x = x
        test_y = y
        
        unique_x = np.unique(train_x)
        unique_y = np.unique(train_y)
        
        if unique_x.size == args.nl and unique_y.size == 5:
            check = True
            print (i)
    
    print (test_x.shape)
    print (test_y.shape)
    """
           
    def train_classifier():
        x = train_x  ### latent reps indexes: 1 D array
        y = train_y ## factor indexes : 1 D array
        
        classifier = np.zeros((z_size,2)) ## stores the majority vote class of 10 latent reps
        for i in range(z_size):
            idx_i = np.where(x == i)
            outputs_i = y[idx_i]
            unique, counts = np.unique(outputs_i, return_counts=True)
            max_idx = np.argmax(counts, axis=0)
            max_class = unique[max_idx]
            classifier[i,0] = i
            classifier[i,1] = max_class
        
        return classifier
    
    def test_classifier():
        x = train_x
        y = train_y
        count = 0
        for i in range(x.shape[0]):
            x_i = x[i]
            #print (x_i)
            y_pred = classifier[x_i.astype(int),1]
            if y_pred== y[i]:
                count+=1
        return (count)
    
    
    
    classifier = train_classifier()
    print (classifier)
    filep.write(str(epoch)+'\n')
    filep.write(str(classifier))
    filep.write('\n')
    count = test_classifier()
    print count
    filep.write(str(count))
    filep.write('\n')
    




def train(args,lrate):
    
    print ("Copying the dataset to the current node's  dir...")
    
    tmp='/Tmp/vermavik/'
    home='/u/vermavik/'
    
    dataset = 'dsprite'
    data_source_dir = home+'data/'+dataset+'/'
    
    ### set up the experiment directories########
    exp_name = experiment_name_lwb(train_size= args.train_size,
                    meta_steps = args.meta_steps,
                    sigma = args.sigma,
                    temperature_factor = args.temperature_factor,
                    alpha1 = args.alpha1,
                    alpha2 = args.alpha2,
                    alpha3 = args.alpha3,
                    grad_max_norm = args.grad_max_norm,
                    epochs= args.epochs,
                    z_size = args.nl,
                    transition_size = args.transition_size,
                    transition_steps = args.transition_steps, 
                    job_id=None,
                    add_name='')
    #temp_model_dir = tmp+'experiments/HVWB/'+dataset+'/model/'+ exp_name
    #temp_result_dir = tmp+'experiments/HVWB/'+dataset+'/results/'+ exp_name
    model_dir = home+'experiments/HVWB/'+dataset+'/model/'+ exp_name
    result_dir = home+'experiments/HVWB/'+dataset+'/results/'+ exp_name
    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    
    result_path = os.path.join(result_dir , 'out.txt')
    bufsize = 0
    filep = open(result_path, 'w', bufsize)
    
    out_str = str(args)
    print(out_str)
    filep.write(out_str + '\n') 
    
   
    for batch_idx, data in enumerate(train_loader):
        Xbatch = data.numpy()
        scl = 1./np.sqrt(np.mean((Xbatch-np.mean(Xbatch))**2))
        shft = -np.mean(Xbatch*scl)
        break ### TO DO : calculate statistics on whole data
    
    model = Net_LWB(args)
    
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
    print  'Number of steps....'
    print args.num_steps
    print "Number of metasteps...."
    print args.meta_steps
    
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
        for batch_idx, data in enumerate(train_loader):
            data = data.unsqueeze(1)
            if args.cuda:
                data = data.cuda()
            data = Variable(data)
            
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
                    #data, _, _, _, _, _, _ = forward_diffusion(data, model, loss_fn,temperature_forward,meta_step)
                    #data = data.view(-1,3, 64,64)
                    #data = Variable(data.data, requires_grad=False)
                    x =  Variable(x_tilde.data.view(-1, 1, 64, 64), requires_grad=False)
                    z = Variable(z_tilde.data, requires_grad=False)
                    if args.encode_every_step==0:
                        encode = False
                    temperature_forward *= args.temperature_factor
                    
            if batch_idx%1000==0:
                plot_loss(model_dir, train_loss, train_x_loss, train_log_p_reverse, train_kld, train_loss_each_step, train_x_loss_each_step, train_log_p_reverse_each_step, args.meta_steps)
            
            if np.isnan(loss.data.cpu()[0]) or np.isinf(loss.data.cpu()[0]):
                print loss.data
                print 'NaN detected'
                return 1.
            
        
        torch.save(model, model_dir+'/model.pt')
        #for i in range(args.num_steps*args.meta_steps):
        #    get_disentanglement_score(model_dir, step=i+1)
        get_disentanglement_score(epoch, model_dir, filep, step=1) ## step index starts from 1
    
    
if __name__ == '__main__':
    args = parse_args()
    train(args, lrate=0.0001)
    pass

