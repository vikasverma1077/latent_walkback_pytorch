'''
Created on Apr 15, 2018

@author: vermavik
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.distributions import log_normal2



class Net_LWB(nn.Module):
    def __init__(self, args, input_shape=(1,64,64)):
        super(Net_LWB, self).__init__()
        
        self.args = args
        kernel_size = 4
        padsize = 2
        
        self.act = nn.ReLU()
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
        
        self.fc_trans_1 = nn.Linear(args.nl, self.args.transition_size)
        self.transition_params.extend(self.fc_trans_1.parameters())
        self.bn8_list=nn.ModuleList()
        #print args.meta_steps
        for i in xrange(args.meta_steps):
            self.bn8_list.append(nn.BatchNorm1d(self.args.transition_size))
            self.transition_params.extend(self.bn8_list[i].parameters())
            #print _
       
        self.fc_trans_1_1 = nn.Linear(self.args.transition_size, self.args.transition_size)
        self.transition_params.extend(self.fc_trans_1_1.parameters())
        self.bn9_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn9_list.append(nn.BatchNorm1d(self.args.transition_size))
            self.transition_params.extend(self.bn9_list[i].parameters())
        
        self.fc_trans_1_2 = nn.Linear(self.args.transition_size, self.args.transition_size)
        self.transition_params.extend(self.fc_trans_1_2.parameters())
        self.bn10_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn10_list.append(nn.BatchNorm1d(self.args.transition_size))
            self.transition_params.extend(self.bn10_list[i].parameters())
            
        self.fc_trans_1_3 = nn.Linear(self.args.transition_size, self.args.transition_size)
        self.transition_params.extend(self.fc_trans_1_3.parameters())
        self.bn10_1_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn10_1_list.append(nn.BatchNorm1d(self.args.transition_size))
            self.transition_params.extend(self.bn10_1_list[i].parameters())
            
            
        self.fc_trans_1_4 = nn.Linear(self.args.transition_size, 128)
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
      
        if self.args.cuda:
            sigma_ = Variable(torch.sqrt(torch.FloatTensor(1).fill_(self.args.sigma * temperature)).cuda())
            #print ('sigma_', np.isnan(sigma_.data.cpu().numpy()).any())
        
        else:
            sigma_ = Variable(torch.sqrt(torch.FloatTensor(1).fill_(self.args.sigma * temperature)))
        
        z_new = eps.mul(sigma.mul(0.5).exp_()).mul(sigma_).add_(mu)
        #print ('z_new', np.isnan(z_new.data.cpu().numpy()).any())
        z_new = (z_new - z_new.mean(0))/(0.001+ z_new.std(0))
        #print ('z_new_mean', np.isnan(z_new.mean(0).data.cpu().numpy()).any())
        #print ('z_new_std', np.isnan(z_new.std(0).data.cpu().numpy()).any())
        #print ('z_new', np.isnan(z_new.data.cpu().numpy()).any())
        
        
       
        if self.args.cuda:
            sigma_ = Variable(torch.log(torch.FloatTensor(1).fill_(self.args.sigma * temperature)).cuda()) + sigma
            #print ('sigma2', np.isnan(sigma_.data.cpu().numpy()).any())
        else:
            sigma_ = Variable(torch.log(torch.FloatTensor(1).fill_(self.args.sigma * temperature))) + sigma
        
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


class VAE(nn.Module):
    def __init__(self, args, imgSize=(1,64,64)):
        super(VAE, self).__init__()
        
        self.args = args
        
        ## encoder###
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3= nn.Conv2d(32, 64, 4, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.flat_shape = self.get_flat_shape(imgSize)
        
        self.fc1 = nn.Linear(self.flat_shape, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc21 = nn.Linear(128, self.args.nl)
        self.bn5_1 = nn.BatchNorm1d(self.args.nl)
        self.fc22 = nn.Linear(128, self.args.nl)
        self.bn5_2 = nn.BatchNorm1d(self.args.nl)
        ### decoder###
        self.fc3 = nn.Linear(self.args.nl, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 8*8*64)
        self.bn7 = nn.BatchNorm1d(8*8*64)
        
        self.conv_z_1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride= 2, padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv_z_2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride= 2, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.conv_z_3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride= 2, padding=1)
        self.bn10 = nn.BatchNorm2d(1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.imgSize = imgSize
    
    def get_flat_shape(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.conv1(dummy)
        dummy = self.conv2(dummy)
        dummy = self.conv3(dummy)
        dummy = self.conv4(dummy)
        return dummy.data.view(1, -1).size(1)
    
    def encode(self, x):
        #x = x.view(-1,1,64,64)
        h1 = self.relu(self.bn1(self.conv1(x)))
        h2 = self.relu(self.bn2(self.conv2(h1)))
        h3 = self.relu(self.bn3(self.conv3(h2)))
        h4 = self.relu(self.bn4(self.conv4(h3)))
        #print (h4.shape)
        h4 = h4.view(-1, self.flat_shape)
        #print (h4.shape)
        h5 = self.relu(self.bn5(self.fc1(h4)))
        return self.bn5_1(self.fc21(h5)), self.bn5_2(self.fc22(h5))

    def reparameterize(self, mu, logvar):
        #if self.training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
        #else:
        #return mu

    def decode(self, z):
        #print (z.shape)
        h5 = self.relu(self.bn6(self.fc3(z)))
        h6 = self.relu(self.bn7(self.fc4(h5)))
        h6 = h6.view(-1, 64, 8,8)
        #print (h6.shape)
        h7 = self.relu(self.bn8(self.conv_z_1(h6)))
        #print (h7.min())#, h7.min())
        h8 = self.relu(self.bn9(self.conv_z_2(h7)))
        #print (h8.max())
        #h9 = self.relu(self.conv_z_3(h8))
        return self.sigmoid(self.bn10(self.conv_z_3(h8)))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, *(self.imgSize)))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

