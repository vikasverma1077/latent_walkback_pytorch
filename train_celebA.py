import argparse
import numpy as np
import os
import mimir
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim


from viz import plot_images
import sys
#from lib.util import  norm_weight, _p, itemlist,  load_params, create_log_dir, unzip,  save_params
from lib.distributions import log_normal2

from load import *
from distutils.dir_util import copy_tree
from shutil import rmtree
from collections import OrderedDict


sys.setrecursionlimit(10000000)
INPUT_SIZE = 3*64*64
WIDTH=64
N_COLORS=3
use_conv = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=500, type=int,
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
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Name of dataset to use.')
    parser.add_argument('--data_aug', type=int, default=0)
    parser.add_argument('--plot_before_training', type=bool, default=False,
                        help='Save diagnostic plots at epoch 0, before any training.')
    parser.add_argument('--num_steps', type=int, default=2,
                        help='Number of transition steps.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Standard deviation of the diffusion process.')
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
    parser.add_argument('--temperature_factor', type = float, default = 2.0,
                        help='How much temperature must be scaled')
    parser.add_argument('--sigma', type = float, default = 0.00001,
                        help='How much Noise should be added at step 1')
    parser.add_argument('--use_decoder', type = bool, default = True,
                        help='whether should we use decoder')
    parser.add_argument('--use_encoder', type = bool, default = True,
                        help='whether should we use encoder')
    parser.add_argument('--nl', type = int, default = 512,
                        help='Size of Latent Size')
    
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model_args = eval('dict(' + args.model_args + ')')
    print model_args


    if not os.path.exists(args.output_dir):
        raise IOError("Output directory '%s' does not exist. "%args.output_dir)
    return args, model_args





hw_size=64  ## height and width of MNIST data

class Net(nn.Module):
    def __init__(self, args,input_shape=(3,hw_size,hw_size)):
        super(Net, self).__init__()
        
        kernel_size = 5
        padsize = 2
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        ### Encoder ######
               
        self.conv_x_z_1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=2, padding=padsize)
        self.bn1 = nn.BatchNorm2d(64) 
        self.conv_x_z_2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=padsize)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv_x_z_3 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=2, padding=padsize)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.flat_shape = self.get_flat_shape_1(input_shape) 
        self.fc_layer_1 = nn.Linear(self.flat_shape, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        
        self.fc_z_mu = nn.Linear(1024, args.nl)
        self.fc_z_sigma = nn.Linear(1024, args.nl)
        
       
        ###### transition operator ########
        self.fc_trans_1 = nn.Linear(args.nl, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc_trans_1_1 = nn.Linear(1024, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.fc_trans_1_2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        
               
        ### decoder #####
        self.fc_z_x_1 = nn.Linear(args.nl, 256*8*8)
        self.bn8 = nn.BatchNorm1d(256*8*8)
        self.conv_z_x_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride= 2, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv_z_x_3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride= 2, padding=1)
        self.bn10 = nn.BatchNorm2d(64)
        self.conv_z_x_4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride= 2, padding=1)
        self.bn11 = nn.BatchNorm2d(3)
        
           
            
    def get_flat_shape_1(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.conv_x_z_1(dummy)
        dummy = self.conv_x_z_2(dummy)
        dummy = self.conv_x_z_3(dummy)
        
        return dummy.data.view(1, -1).size(1)
    
    
    def get_flat_shape_2(self, shape):
        dummy = Variable(torch.zeros(1, *shape))
        dummy = self.conv_z_x_2(dummy)
        dummy = self.conv_z_x_3(dummy)
        dummy = self.conv_z_x_4(dummy)
        
        return dummy.data.shape
    
    def encode(self, x):
        
        c1 = self.relu(self.bn1(self.conv_x_z_1(x)))
        c2 = self.relu(self.bn2(self.conv_x_z_2(c1)))
        c3 = self.relu(self.bn3(self.conv_x_z_3(c2)))
        
        h1 = c3.view(-1, self.flat_shape)
        h1 = self.relu(self.bn4(self.fc_layer_1(h1)))
        
        mu = self.relu(self.fc_z_mu(h1))
        logvar = self.relu(self.fc_z_sigma(h1))
        
        return mu, logvar

        
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    
       
    def transition (self, z, temperature):
        
        h1 = self.relu(self.bn5(self.fc_trans_1(z)))
        h2 = self.relu(self.bn6(self.fc_trans_1_1(h1)))
        h3 = self.relu(self.bn7(self.fc_trans_1_2(h2)))
        h3 = torch.clamp(h3, min=0, max=5)
        
        mu = self.fc_z_mu(h3)
        sigma = self.fc_z_sigma(h3)
        
        eps = Variable(mu.data.new(mu.size()).normal_())
        
        #z_new = mu + T.sqrt(args.sigma * temperature) * T.exp(0.5 * sigma) * eps
        #z_new = (z_new - T.mean(z_new, axis=0, keepdims=True)) / (0.001 + T.std(z_new, axis=0, keepdims=True))
        
        if args.cuda:
            sigma_ = Variable(torch.sqrt(torch.FloatTensor(1).fill_(args.sigma * temperature)).cuda())
        else:
            sigma_ = Variable(torch.sqrt(torch.FloatTensor(1).fill_(args.sigma * temperature)))
        
        z_new = eps.mul(sigma.mul(0.5).exp_()).mul(sigma_).add_(mu)
        z_new = (z_new - z_new.mean(0))/(0.001+ z_new.std(0))
        
       
        if args.cuda:
            sigma_ = Variable(torch.log(torch.FloatTensor(1).fill_(args.sigma * temperature)).cuda()) + sigma
        else:
            sigma_ = Variable(torch.log(torch.FloatTensor(1).fill_(args.sigma * temperature))) + sigma
        
        log_p_reverse = log_normal2(z, mu, sigma_, eps = 1e-6).sum(1).mean()
        z_new = torch.clamp(z_new, min=-4, max=4)
        
        return z_new, log_p_reverse, sigma, h2
        
     
    def decode (self, z_new):
        d0 = self.relu(self.bn8(self.fc_z_x_1(z_new)))
        d0 = d0.view(args.batch_size, 256, 8, 8)
        d1 = self.relu(self.bn9(self.conv_z_x_2(d0)))
        d2 = self.relu(self.bn10(self.conv_z_x_3(d1)))
        d3 = self.sigmoid(self.conv_z_x_4(d2))
        shape = d3.data.shape
        p =  d3.view(-1, shape[1]*shape[2]*shape[3])
        
        eps = 1e-4
        p = torch.clamp(p, min= eps, max=1.0 - eps)
        #x_loss =  -T.nnet.binary_crossentropy(p, x).sum(axis=1)
        return p
    
    def sample(self, z, temperature):
        d0 = self.relu(self.bn4(self.fc_z_x_1(z)))
        d0 = d0.view(args.batch_size, 32, 7, 7)
        d1 = self.relu(self.bn5(self.conv_z_x_2(d0)))
        d2 = self.sigmoid(self.bn6(self.conv_z_x_3(d1)))
        #d3 = self.sigmoid(self.conv_z_x_4(d2))
        x_new =  d2.view(-1, self.get_flat_shape(input_shape=(258,8,8)))
    
        z_new, log_p_reverse, sigma, h2 = self.transition( x_new , temperature)
        x_tilde = self.decode(z_new)
        
        return x_tilde, x_new, z_new 
        


def compute_loss(x, model, loss_fn, start_temperature):
    temperature = start_temperature
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    z_tilde, log_p_reverse, sigma, h2 = model.transition( z, temperature)
    x_tilde = model.decode(z_tilde)
    x_loss = loss_fn(x_tilde,x.view(-1, 3*64*64))## sum over axis=1
    #print x_loss.data.shape
    loss = log_p_reverse + x_loss
    x_states = [x_tilde]
    z_states = [z_tilde]
     
    total_loss = loss
    
    #print args.num_steps
    for _ in range(args.num_steps - 1):
        temperature *= args.temperature_factor
        x = Variable(x_tilde.data, requires_grad=False)
        mu, logvar = model.encode(x_tilde.view(-1, 3, 64, 64))
        z = model.reparameterize(mu, logvar)
        z_tilde, log_p_reverse, sigma, h2 = model.transition( z, temperature)
        x_tilde = model.decode(z_tilde)
        x_loss = loss_fn(x_tilde,x)## sum over axis=1
        loss = log_p_reverse + x_loss
        x_states.append(x_tilde)
        z_states.append(z_tilde)
        total_loss = total_loss+ loss
     #loss = -T.mean(sum(log_p_reverse_list, 0.0))
    loss = - total_loss/args.num_steps
    return loss



def forward_diffusion(x, model, loss_fn,temperature):
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    z_tilde, log_p_reverse, sigma, h2 = model.transition( z, temperature)
    x_tilde = model.decode(z_tilde)
    x_loss = loss_fn (x_tilde,x)## sum over axis=1
    total_loss = log_p_reverse + x_loss
    
    return x_tilde, total_loss, z_tilde, x_loss, log_p_reverse, sigma, h2



##############################


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
    dataset='celebA'
    data_source_dir = home+'data/CelebA/'
    """
    if not os.path.exists(data_source_dir):
        os.makedirs(data_source_dir)
    data_target_dir = tmp+'data/CelebA/'
    copy_tree(data_source_dir, data_target_dir)
    """
    ### set up the experiment directories########
    """
    exp_name=experiment_name(args.epochs,
                    args.batch_size,
                    args.test_batch_size,
                    args.lr,
                    args.momentum, 
                    args.alpha1,
                    args.alpha2,
                    args.alpha3,
                    args.data_aug,
                    args.job_id,
                    args.add_name)
    """
    exp_name = 'temp'
    temp_model_dir = tmp+'experiments/HVWB/'+dataset+'/model/'+ exp_name
    temp_result_dir = tmp+'experiments/HVWB/'+dataset+'/results/'+ exp_name
    model_dir = home+'experiments/HVWB/'+dataset+'/model/'+ exp_name
    result_dir = home+'experiments/HVWB/'+dataset+'/results/'+ exp_name
    
    
    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)
    
    if not os.path.exists(temp_result_dir):
        os.makedirs(temp_result_dir)
    """   
    #copy_script_to_folder(os.path.abspath(__file__), temp_result_dir)
    result_path = os.path.join(temp_result_dir , 'out.txt')
    filep = open(result_path, 'w')
    
    out_str = str(args)
    print(out_str)
    filep.write(out_str + '\n') 
    
      
    #torch.backends.cudnn.enabled = False # slower but repeatable
    torch.backends.cudnn.enabled = True # faster but not repeatable
                      
    out_str = 'initial seed = ' + str(args.manualSeed)
    print(out_str)
    filep.write(out_str + '\n\n')
    """
    #model_id = '/data/lisatmp4/anirudhg/minst_walk_back/walkback_'
    
    """
    model_id = '/data/lisatmp4/anirudhg/celebA_latent_walkback/walkback_'
    model_dir = create_log_dir(args, model_id)
    model_id2 =  '../celebA_logs/walkback_'
    model_dir2 = create_log_dir(args, model_id2)
    print model_dir
    print model_dir2 + '/' + 'log.jsonl.gz'
    logger = mimir.Logger(filename=model_dir2  + '/log.jsonl.gz', formatter=None)
    """
    # TODO batches_per_epoch should not be hard coded
    lrate = args.lr
    import sys
    sys.setrecursionlimit(10000000)
    args, model_args = parse_args()
    print args

   
    ## load the training data
    
    print 'loading celebA'
    #train_loader, test_loader=load_celebA(args.data_aug, args.batch_size, args.batch_size,args.cuda, data_source_dir)
    n_colors = 3
    spatial_width = 64

    

    print "Width", WIDTH, spatial_width
    
    
    model = Net(args)
    if args.cuda:
        model.cuda()
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)
    
    uidx = 0
    estop = False
    bad_counter = 0
    max_epochs = 1
    batch_index = 1
    n_samples = 0
    print  'Number of steps....'
    print args.num_steps
    print "Number of metasteps...."
    print args.meta_steps
    print 'Done'
    count_sample = 1
   
    
    for epoch in range(max_epochs):
        
        #for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = torch.randn(500,3,64,64)
            target = torch.randn(500,1)
            print (data.shape)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            t0 = time.time()
            batch_index += 1
            n_samples += data.data.shape[0]
            #print (n_samples)
            temperature_forward = args.temperature
            meta_cost = []
            for meta_step in range(0, args.meta_steps):
                loss= compute_loss(data, model, loss_fn, temperature_forward)
                    #meta_cost.append(loss)
                print loss
                loss.backward()
            
            optimizer.step()
            
            #cost = sum(meta_cost) / len(meta_cost)
            #print cost
            #gradient_updates_ = get_grads(data_use[0],args.temperature)
            """
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN detected'
                return 1.
            """
            
            
            
    
    copy_tree(temp_model_dir, model_dir)
    copy_tree(temp_result_dir, result_dir)
    
    rmtree(temp_model_dir)
    rmtree(temp_result_dir)
    

if __name__ == '__main__':
    args, model_args = parse_args()
    train(args, model_args, lrate=0.000001)
    pass

