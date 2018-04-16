## code merged from train_celebA_diff_bnorm_v2.py and disentanglement_dsprite_vae_kim_conv.py

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


from viz import plot_images
import sys
#from lib.util import  norm_weight, _p, itemlist,  load_params, create_log_dir, unzip,  save_params
from lib.distributions import log_normal2

from load import *
from distutils.dir_util import copy_tree
from shutil import rmtree
from collections import OrderedDict


sys.setrecursionlimit(10000000)
INPUT_SIZE = 1*64*64
WIDTH=64
N_COLORS=1
use_conv = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, default='leakyrelu',
                        help='activation function to use in the network except the last layer of decoder')
    parser.add_argument('--encode_every_step', type=int, default=1)
    parser.add_argument('--alpha1', type=float, default=1.0,help='coefficient for reconstruction loss')
    parser.add_argument('--alpha2', type=float, default=1.0,help='coefficient for log_p_reverse')
    parser.add_argument('--alpha3', type=float, default=1.0,help='coefficient for KLD')
    
    
    parser.add_argument('--epochs', type = int, default = 10,
                        help='num of epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Initial learning rate. ' + \
                        'Will be decayed until it\'s 1e-5.')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--resume_file', default=None, type=str,
                        help='Name of saved model to continue training')
    parser.add_argument('--suffix', default='', type=str,
                        help='Optional descriptive suffix for model')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='Output directory to store trained models')
    parser.add_argument('--ext-every-n', type=int, default=25,
                        help='Evaluate training extensions every N epochs')
    parser.add_argument('--model-args', type=str, default='',
                        help='Dictionary string to be eval()d containing model arguments.')
    parser.add_argument('--dropout_rate', type=float, default=0.,
                        help='Rate to use for dropout during training+testing.')
    parser.add_argument('--dataset', type=str, default='dsprite',
                        help='Name of dataset to use.')
    parser.add_argument('--data_aug', type=int, default=0)
    parser.add_argument('--plot_before_training', type=bool, default=False,
                        help='Save diagnostic plots at epoch 0, before any training.')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='Number of transition steps.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Standard deviation of the diffusion process.')
    
    parser.add_argument('--grad_max_norm', type=float, default=5.0,
                        help='max value of grad norm used for gradient clipping')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha factor')
    parser.add_argument('--dims', default=[4096], type=int,
                        nargs='+')
    parser.add_argument('--noise_prob', default=0.1, type=float,
                        help='probability for bernouli distribution of adding noise of 1 to each input')
    parser.add_argument('--avg', default=0, type=float)
    parser.add_argument('--std', default=1., type=float)
    parser.add_argument('--noise', default='gaussian', choices=['gaussian', 'binomial'])
    parser.add_argument('--reload_', type=bool, default = False,
                        help='Reloading the parameters')
    parser.add_argument('--saveto_filename', type = str, default = None,
                        help='directory where parameters are stored')
    parser.add_argument('--extra_steps', type = int, default = 10,
                        help='Number of extra steps to sample at temperature 1')
    parser.add_argument('--meta_steps', type = int, default = 10,
                        help='Number of extra steps to sample at temperature 1')
    parser.add_argument('--optimizer', type = str, default = 'sgd',
                        help='optimizer we are going to use!!')
    parser.add_argument('--temperature_factor', type = float, default = 1.1,
                        help='How much temperature must be scaled')
    parser.add_argument('--sigma', type = float, default = 0.00001,
                        help='How much Noise should be added at step 1')
   
    parser.add_argument('--use_decoder', type = bool, default = True,
                        help='whether should we use decoder')
    parser.add_argument('--use_encoder', type = bool, default = True,
                        help='whether should we use encoder')
    parser.add_argument('--nl', type = int, default = 10,
                        help='Size of Latent Size')
    parser.add_argument('--job_id', type=str, default='')
    parser.add_argument('--add_name', type=str, default='')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model_args = eval('dict(' + args.model_args + ')')
    print model_args


    if not os.path.exists(args.output_dir):
        raise IOError("Output directory '%s' does not exist. "%args.output_dir)
    return args, model_args


##############################
def experiment_name(dataset='dsprite',
                    act = 'relu',
                    meta_steps=10,
                    sigma = 0.0001,
                    temperature_factor = 1.1,
                    alpha1 = 1.0,
                    alpha2 = 0.01,
                    alpha3 = 1.0,
                    grad_norm_max = 5.0,
                    epochs=10,
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
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)
    
    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name



def plot_loss(exp_dir, train_loss,  train_x_loss, train_log_p_reverse, train_kld,
             train_loss_each_step, train_x_loss_each_step, train_log_p_reverse_each_step, meta_steps):
    
    ### plot metrics from all the steps in one plot###
    plt.plot(np.asarray(train_loss), label='train_loss')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_loss.png' ))
    plt.clf()
    
    plt.plot(np.asarray(train_x_loss), label='train_x_loss')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_x_loss.png' ))
    plt.clf()
    
    plt.plot(np.asarray(train_log_p_reverse), label='train_log_p_reverse')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_log_p_reverse.png' ))
    plt.clf()

    plt.plot(np.asarray(train_kld), label='train_kld')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_kld.png' ))
    plt.clf()
    
    
    ### plot metrics from different steps in different plots##
    for i in range(meta_steps):
    
        plt.plot(np.asarray(train_loss_each_step[i]), label='train_loss')
            
        #plt.ylim(0,2000)
        plt.xlabel('evaluation step')
        plt.ylabel('metrics')
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(exp_dir, 'train_loss'+str(i)+'.png' ))
        plt.clf()
    
    for i in range(meta_steps):
        
        plt.plot(np.asarray(train_x_loss_each_step[i]), label='train_x_loss')
            
        #plt.ylim(0,2000)
        plt.xlabel('evaluation step')
        plt.ylabel('metrics')
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(exp_dir, 'train_x_loss'+str(i)+'.png' ))
        plt.clf()
        
    for i in range(meta_steps):
        plt.plot(np.asarray(train_log_p_reverse_each_step[i]), label='train_log_p_reverse')
            
        #plt.ylim(0,2000)
        plt.xlabel('evaluation step')
        plt.ylabel('metrics')
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(exp_dir, 'train_log_p_reverse'+str(i)+'.png' ))
        plt.clf()

    
    
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



def reverse_time(scl, shft, sample_drawn, name):
    new_image = np.asarray(sample_drawn).astype('float32').reshape(args.batch_size, N_COLORS, WIDTH, WIDTH)
    plot_images(new_image, name)



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

args, model_args = parse_args()

# Sample latents randomly
latents_train = sample_latent(size=700000)
latents_test = sample_latent(size=1000)

# Select images
imgs_train = imgs[latent_to_index(latents_train)]
imgs_test = imgs[latent_to_index(latents_test)]

train_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_train).type(torch.FloatTensor))
test_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_test).type(torch.FloatTensor))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)



hw_size=64  ## height and width of MNIST data

class Net(nn.Module):
    def __init__(self, args,input_shape=(1,hw_size,hw_size)):
        super(Net, self).__init__()
        
        kernel_size = 4
        padsize = 2
        
        if args.activation == 'relu':
            self.act = nn.ReLU()
        elif args.activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        ### Encoder ######
        self.encoder_params= []   
         
        self.conv_x_z_1 = nn.Conv2d(1, 32, kernel_size=kernel_size, stride=2)
        self.encoder_params.extend(self.conv_x_z_1.parameters())
        self.bn1_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn1_list.append(nn.BatchNorm2d(32))
            self.encoder_params.extend(self.bn1_list[i].parameters())
        
        self.conv_x_z_2 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=2)
        self.encoder_params.extend(self.conv_x_z_2.parameters())
        self.bn2_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn2_list.append(nn.BatchNorm2d(32))
            self.encoder_params.extend(self.bn2_list[i].parameters())
        
        self.conv_x_z_3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=2)
        self.encoder_params.extend(self.conv_x_z_3.parameters())
        self.bn3_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn3_list.append(nn.BatchNorm2d(64))
            self.encoder_params.extend(self.bn3_list[i].parameters())
            
        self.conv_x_z_4 = nn.Conv2d(64, 64, kernel_size=kernel_size)
        self.encoder_params.extend(self.conv_x_z_4.parameters())
        self.bn4_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn4_list.append(nn.BatchNorm2d(64))
            self.encoder_params.extend(self.bn4_list[i].parameters())
        
        
        
        
        self.flat_shape = self.get_flat_shape_1(input_shape) 
        self.fc_layer_1 = nn.Linear(self.flat_shape, 128)
        self.encoder_params.extend(self.fc_layer_1.parameters())
        self.bn5_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn5_list.append(nn.BatchNorm1d(128))
            self.encoder_params.extend(self.bn5_list[i].parameters())
            
        self.fc_z_mu = nn.Linear(128, args.nl)
        self.encoder_params.extend(self.fc_z_mu.parameters())
        self.bn6_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn6_list.append(nn.BatchNorm1d(args.nl))
            self.encoder_params.extend(self.bn6_list[i].parameters())
        
        self.fc_z_sigma = nn.Linear(128, args.nl)
        self.encoder_params.extend(self.fc_z_sigma.parameters())
        self.bn7_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn7_list.append(nn.BatchNorm1d(args.nl))
            self.encoder_params.extend(self.bn7_list[i].parameters())
        
       
        ###### transition operator ########
        self.transition_params = []
        
        self.fc_trans_1 = nn.Linear(args.nl, 128)
        self.transition_params.extend(self.fc_trans_1.parameters())
        self.bn8_list=nn.ModuleList()
        #print args.meta_steps
        for i in xrange(args.meta_steps):
            self.bn8_list.append(nn.BatchNorm1d(128))
            self.transition_params.extend(self.bn8_list[i].parameters())
            #print _
       
        self.fc_trans_1_1 = nn.Linear(128, 128)
        self.transition_params.extend(self.fc_trans_1_1.parameters())
        self.bn9_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn9_list.append(nn.BatchNorm1d(128))
            self.transition_params.extend(self.bn9_list[i].parameters())
        
        self.fc_trans_1_2 = nn.Linear(128, 128)
        self.transition_params.extend(self.fc_trans_1_2.parameters())
        self.bn10_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn10_list.append(nn.BatchNorm1d(128))
            self.transition_params.extend(self.bn10_list[i].parameters())
            
        self.fc_trans_1_3 = nn.Linear(128, 128)
        self.transition_params.extend(self.fc_trans_1_3.parameters())
        self.bn10_1_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn10_1_list.append(nn.BatchNorm1d(128))
            self.transition_params.extend(self.bn10_1_list[i].parameters())
            
            
        self.fc_trans_1_4 = nn.Linear(128, 128)
        self.transition_params.extend(self.fc_trans_1_4.parameters())
        self.bn10_2_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn10_2_list.append(nn.BatchNorm1d(128))
            self.transition_params.extend(self.bn10_2_list[i].parameters())
        
               
        ### decoder #####
        self.decoder_params = []
        
        self.fc_z_x_1 = nn.Linear(args.nl, 128)
        self.decoder_params.extend(self.fc_z_x_1.parameters())
        self.bn11_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn11_list.append(nn.BatchNorm1d(128))
            self.decoder_params.extend(self.bn11_list[i].parameters())
            
        self.fc_z_x_2 = nn.Linear(128, 8*8*64)
        self.decoder_params.extend(self.fc_z_x_2.parameters())
        self.bn12_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn12_list.append(nn.BatchNorm1d(8*8*64))
            self.decoder_params.extend(self.bn12_list[i].parameters())
        
            
        self.conv_z_x_1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride= 2, padding=1)
        self.decoder_params.extend(self.conv_z_x_1.parameters())
        self.bn13_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn13_list.append(nn.BatchNorm2d(64))
            self.decoder_params.extend(self.bn13_list[i].parameters())
            
            
        self.conv_z_x_2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride= 2, padding=1)
        self.decoder_params.extend(self.conv_z_x_2.parameters())
        self.bn14_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn14_list.append(nn.BatchNorm2d(32))
            self.decoder_params.extend(self.bn14_list[i].parameters())
            
            
        self.conv_z_x_3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride= 2, padding=1)
        self.decoder_params.extend(self.conv_z_x_3.parameters())
        self.bn15_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn15_list.append(nn.BatchNorm2d(1))
            self.decoder_params.extend(self.bn15_list[i].parameters())
            
           
            
    def get_flat_shape_1(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.conv_x_z_1(dummy)
        dummy = self.conv_x_z_2(dummy)
        dummy = self.conv_x_z_3(dummy)
        dummy = self.conv_x_z_4(dummy)
        return dummy.data.view(1, -1).size(1)
    
    
    def get_flat_shape_2(self, shape):
        dummy = Variable(torch.zeros(1, *shape))
        dummy = self.conv_z_x_2(dummy)
        dummy = self.conv_z_x_3(dummy)
        dummy = self.conv_z_x_4(dummy)
        
        return dummy.data.shape
    
    def encode(self, x, step):
        
        c1 = self.act(self.bn1_list[step](self.conv_x_z_1(x)))
        #print c1
        c2 = self.act(self.bn2_list[step](self.conv_x_z_2(c1)))
        #print c2
        c3 = self.act(self.bn3_list[step](self.conv_x_z_3(c2)))
        #print c3
        c4 = self.act(self.bn4_list[step](self.conv_x_z_4(c3)))
        
        h1 = c4.view(-1, self.flat_shape)
        h1 = self.act(self.bn5_list[step](self.fc_layer_1(h1)))
        #print h1
        mu = self.bn6_list[step](self.fc_z_mu(h1))
        sigma = self.bn7_list[step](self.fc_z_sigma(h1))
        return mu, sigma

        
        
    def reparameterize(self, mu, sigma):
        std = sigma.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    
       
    def transition (self, z, temperature, step):
        #print ('z', np.isnan(z.data.cpu().numpy()).any())
        #    print z.requires_grad
        h1 = self.act(self.bn8_list[step](self.fc_trans_1(z)))
        #print h1
        h2 = self.act(self.bn9_list[step](self.fc_trans_1_1(h1)))
        #print h2
        h3 = self.act(self.bn10_list[step](self.fc_trans_1_2(h2)))
        h4 = self.act(self.bn10_1_list[step](self.fc_trans_1_3(h3)))
        h5 = self.act(self.bn10_2_list[step](self.fc_trans_1_4(h4)))
        
        #print h3
        h5 = torch.clamp(h5, min=0, max=5)
        #print h3
        
        mu = self.bn6_list[step](self.fc_z_mu(h5))  #### why not non-linearity applied here
        #print mu
        sigma = self.bn7_list[step](self.fc_z_sigma(h5))
        #print sigma
        #print ('mu', np.isnan(mu.data.cpu().numpy()).any())
        #print ('sigma', np.isnan(sigma.data.cpu().numpy()).any())
        eps = Variable(mu.data.new(mu.size()).normal_())
        
        #print ('eps', np.isnan(eps.data.cpu().numpy()).any())
        
        
        #print eps
        
        #z_new = mu + T.sqrt(args.sigma * temperature) * T.exp(0.5 * sigma) * eps
        #z_new = (z_new - T.mean(z_new, axis=0, keepdims=True)) / (0.001 + T.std(z_new, axis=0, keepdims=True))
        
        if args.cuda:
            sigma_ = Variable(torch.sqrt(torch.FloatTensor(1).fill_(args.sigma * temperature)).cuda())
            #print ('sigma_', np.isnan(sigma_.data.cpu().numpy()).any())
        
        else:
            sigma_ = Variable(torch.sqrt(torch.FloatTensor(1).fill_(args.sigma * temperature)))
        
        z_new = eps.mul(sigma.mul(0.5).exp_()).mul(sigma_).add_(mu)
        #print ('z_new', np.isnan(z_new.data.cpu().numpy()).any())
        z_new = (z_new - z_new.mean(0))/(0.001+ z_new.std(0))
        #print ('z_new_mean', np.isnan(z_new.mean(0).data.cpu().numpy()).any())
        #print ('z_new_std', np.isnan(z_new.std(0).data.cpu().numpy()).any())
        #print ('z_new', np.isnan(z_new.data.cpu().numpy()).any())
        
        
       
        if args.cuda:
            sigma_ = Variable(torch.log(torch.FloatTensor(1).fill_(args.sigma * temperature)).cuda()) + sigma
            #print ('sigma2', np.isnan(sigma_.data.cpu().numpy()).any())
        
        else:
            sigma_ = Variable(torch.log(torch.FloatTensor(1).fill_(args.sigma * temperature))) + sigma
        
        log_p_reverse = log_normal2(z, mu, sigma_, eps = 1e-6).mean()
        #print ('z', np.isnan(z.data.cpu().numpy()).any())
        #print ('log_p_reverse', log_p_reverse)
        z_new = torch.clamp(z_new, min=-4, max=4)
        #print z_new 
        return z_new, log_p_reverse, mu, sigma
        
     
    def decode (self, z_new, step):
        #print z_new
        d0 = self.act(self.bn11_list[step](self.fc_z_x_1(z_new)))
        d1 = self.act(self.bn12_list[step](self.fc_z_x_2(d0)))
        #print d0
        d1 = d1.view(-1, 64, 8, 8)
        d1 = self.act(self.bn13_list[step](self.conv_z_x_1(d1)))
        #print d1
        d2 = self.act(self.bn14_list[step](self.conv_z_x_2(d1)))
        #print self.conv_z_x_3(d1)
        d3 = self.sigmoid(self.bn15_list[step](self.conv_z_x_3(d2)))
        #print self.conv_z_x_4(d2)
        shape = d3.data.shape
        p =  d3.view(-1, shape[1]*shape[2]*shape[3])
        
        eps = 1e-4
        p = torch.clamp(p, min= eps, max=1.0 - eps)
        #x_loss =  -T.nnet.binary_crossentropy(p, x).sum(axis=1)
        return p
    
    def sample(self, z, temperature,step):
        d0 = self.act(self.bn11_list[step](self.fc_z_x_1(z)))
        d1 = self.act(self.bn12_list[step](self.fc_z_x_2(d0)))
        #print d0
        d1 = d1.view(-1, 64, 8, 8)
        d1 = self.act(self.bn13_list[step](self.conv_z_x_1(d1)))
        d2 = self.act(self.bn14_list[step](self.conv_z_x_2(d1)))
        d3 = self.sigmoid(self.bn15_list[step](self.conv_z_x_3(d2)))
        shape = d3.data.shape
        x_new =  d3.view(-1, shape[1]*shape[2]*shape[3])
    
        z_new, log_p_reverse, sigma, h2 = self.transition( z , temperature, step)
        x_tilde = self.decode(z_new,step)
        
        return x_tilde, x_new, z_new 
        



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



def get_disentanglement_score(model_dir,step):
    ### disentanglement_dsprite_lwb computation#####
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
        
        #print (mu)
        #print (mu.std(dim=0))
        #print (mu)
        #print (std_each_z)
        normalized_z = mu/std_each_z
        
        #print std_each_z
        #print (normalized_z)
        variance_z = normalized_z.var(dim=0)
        
        #print (variance_z)
        
        variance_z = variance_z.cpu()
        
        x = np.argmin(variance_z)
        
        #print (x)
        
        return x
    
    
    ########################################
    ### create train data for classifier####
    ########################################
    check = False ## check to ensure that train_x has all possible indexes of Z's and train_y has all possible indexes of Factors (5 in case of dsprite)
    while (check== False):
        print ('this')
        num_samples = 1000
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
        
        #print (check)
    #print (train_x.shape)
    #print (train_y.shape)
    
    ### create test data for classifier###
    ######################################
    
    check = False ## check to ensure that train_x has all possible indexes of Z's and train_y has all possible indexes of Factors (5 in case of dsprite)
    while (check== False):
    
        num_samples = 1000
        x = np.zeros(num_samples)
        y = np.zeros(num_samples)## target labels for the disentanglement_dsprite_lwb classifier
        
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
    
    #print (test_x.shape)
    #print (test_y.shape)
    
           
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
    
    classifier = train_classifier()
    
    print (classifier)
    
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
        
        print (count)
    
    train_classifier()
    test_classifier()





def train(args,
          model_args,
          lrate):
    
    print ("Copying the dataset to the current node's  dir...")
    
    tmp='/Tmp/vermavik/'
    home='/u/vermavik/'
    """
    tmp='/tmp/vermav1/'
    home='/u/79/vermav1/unix/'
    """
    
    dataset = args.dataset
    data_source_dir = home+'data/'+dataset+'/'
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
                    job_id=args.job_id,
                    add_name=args.add_name)
    
    #temp_model_dir = tmp+'experiments/HVWB/'+dataset+'/model/'+ exp_name
    #temp_result_dir = tmp+'experiments/HVWB/'+dataset+'/results/'+ exp_name
    model_dir = home+'experiments/HVWB/'+dataset+'/model/'+ exp_name
    result_dir = home+'experiments/HVWB/'+dataset+'/results/'+ exp_name
    
    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    #if not os.path.exists(temp_result_dir):
    #    os.makedirs(temp_result_dir)
    
    # TODO batches_per_epoch should not be hard coded
    lrate = args.lr
    import sys
    sys.setrecursionlimit(10000000)
    args, model_args = parse_args()
    print args

   
    ## load the training data
    
    #print 'loading celebA'
    #train_loader, test_loader=load_celebA(args.data_aug, args.batch_size, args.batch_size, args.cuda, data_source_dir)
    n_colors = 1
    spatial_width = 64
    
    for batch_idx, data in enumerate(train_loader):
    
        Xbatch = data.numpy()
        #print Xbatch
        scl = 1./np.sqrt(np.mean((Xbatch-np.mean(Xbatch))**2))
        shft = -np.mean(Xbatch*scl)
        
        break ### TO DO : calculate statistics on whole data
    
    print "Width", WIDTH, spatial_width
    
    
    model = Net(args)
    if args.cuda:
        model.cuda()
    loss_fn = nn.BCELoss()
    if args.optimizer=='sgd':
        optimizer_encoder = optim.SGD(model.encoder_params, lr=args.lr, momentum=args.momentum, weight_decay=0)
        optimizer_transition = optim.SGD(model.transition_params, lr=args.lr, momentum=args.momentum, weight_decay=0)
        optimizer_decoder = optim.SGD(model.decoder_params, lr=args.lr, momentum=args.momentum, weight_decay=0)
    elif args.optimizer=='adam':
        optimizer_encoder = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer_transition = optim.Adam(model.transition_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer_decoder = optim.Adam(model.decoder_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    uidx = 0
    estop = False
    bad_counter = 0
    batch_index = 1
    n_samples = 0
    print  'Number of steps....'
    print args.num_steps
    print "Number of metasteps...."
    print args.meta_steps
    count_sample = 1
   
    
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
            
            
            t0 = time.time()
            #batch_index += 1
            n_samples += data.data.shape[0]
            #print (n_samples)
            temperature_forward = args.temperature
            meta_cost = []
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
                total_norm = clip_grad_norm(model.parameters(), args.grad_max_norm)
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
                    x =  Variable(x_tilde.data.view(-1, 1, spatial_width, spatial_width), requires_grad=False)
                    z = Variable(z_tilde.data, requires_grad=False)
                    if args.encode_every_step==0:
                        encode = False
                    temperature_forward *= args.temperature_factor
                    
           
                #print loss.data
            #print loss.data    
            
            #cost = sum(meta_cost) / len(meta_cost)
            #print cost
            #gradient_updates_ = get_grads(data_use[0],args.temperature)
            
            if np.isnan(loss.data.cpu()[0]) or np.isinf(loss.data.cpu()[0]):
                print loss.data
                print 'NaN detected'
                return 1.
            
        
        torch.save(model, model_dir+'/model.pt')
        #for i in range(args.num_steps*args.meta_steps):
        #    get_disentanglement_score(model_dir, step=i+1)
        get_disentanglement_score(model_dir, step=1)
    #copy_tree(temp_model_dir, model_dir)
    #copy_tree(temp_result_dir, result_dir)
    
    #rmtree(temp_model_dir)
    #rmtree(temp_result_dir)
    

if __name__ == '__main__':
    args, model_args = parse_args()
    train(args, model_args, lrate=0.0001)
    pass

