'''
Created on Apr 14, 2018

@author: vermavik
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
 
from lib.distributions import log_normal2

"""

class Net(nn.Module):
    def __init__(self, args, input_shape=(3,32,32)):
        super(Net, self).__init__()
        
        self.args = args
        self.init_ch = args.init_ch
        self.input_shape = input_shape
        self.enc_fc_size = args.enc_fc_size
        self.transition_size = args.transition_size
        kernel_size = 5
        padsize = 2
        
        
        if self.args.activation == 'relu':
            self.act = nn.ReLU()
        elif self.args.activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        ### Encoder ######
        self.encoder_params= []   
         
        self.conv_x_z_1 = nn.Conv2d(3, self.init_ch, kernel_size=kernel_size, stride=2, padding=padsize)
        self.encoder_params.extend(self.conv_x_z_1.parameters())
        self.bn1_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn1_list.append(nn.BatchNorm2d(self.init_ch))
            self.encoder_params.extend(self.bn1_list[i].parameters())
        
        self.conv_x_z_2 = nn.Conv2d(self.init_ch, self.init_ch*2, kernel_size=kernel_size, stride=2, padding=padsize)
        self.encoder_params.extend(self.conv_x_z_2.parameters())
        self.bn2_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn2_list.append(nn.BatchNorm2d(self.init_ch*2))
            self.encoder_params.extend(self.bn2_list[i].parameters())
        
        self.conv_x_z_3 = nn.Conv2d(self.init_ch*2, self.init_ch*4, kernel_size=kernel_size, stride=2, padding=padsize)
        self.encoder_params.extend(self.conv_x_z_3.parameters())
        self.bn3_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn3_list.append(nn.BatchNorm2d(self.init_ch*4))
            self.encoder_params.extend(self.bn3_list[i].parameters())
            
        self.flat_shape = self.get_flat_shape_1(input_shape) 
        self.fc_layer_1 = nn.Linear(self.flat_shape, self.enc_fc_size)
        self.encoder_params.extend(self.fc_layer_1.parameters())
        self.bn4_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn4_list.append(nn.BatchNorm1d(self.enc_fc_size))
            self.encoder_params.extend(self.bn4_list[i].parameters())
            
        self.fc_z_mu = nn.Linear(self.enc_fc_size, args.nl)
        self.encoder_params.extend(self.fc_z_mu.parameters())
        self.bn5_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn5_list.append(nn.BatchNorm1d(args.nl))
            self.encoder_params.extend(self.bn5_list[i].parameters())
        
        self.fc_z_sigma = nn.Linear(self.enc_fc_size, args.nl)
        self.encoder_params.extend(self.fc_z_sigma.parameters())
        self.bn6_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn6_list.append(nn.BatchNorm1d(args.nl))
            self.encoder_params.extend(self.bn6_list[i].parameters())
        
       
        ###### transition operator ########
        self.transition_params = []
        
        self.fc_trans_1 = nn.Linear(args.nl, self.transition_size)
        self.transition_params.extend(self.fc_trans_1.parameters())
        self.bn7_list=nn.ModuleList()
        #print args.meta_steps
        for i in xrange(args.meta_steps):
            self.bn7_list.append(nn.BatchNorm1d(self.transition_size))
            self.transition_params.extend(self.bn7_list[i].parameters())
            #print _
       
        self.fc_trans_2 = nn.Linear(self.transition_size, self.transition_size)
        self.transition_params.extend(self.fc_trans_2.parameters())
        self.bn8_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn8_list.append(nn.BatchNorm1d(self.transition_size))
            self.transition_params.extend(self.bn8_list[i].parameters())
        
        if self.args.transition_steps==3:
            self.fc_trans_3 = nn.Linear(self.transition_size, self.enc_fc_size)
            self.transition_params.extend(self.fc_trans_3.parameters())
            self.bn9_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn9_list.append(nn.BatchNorm1d(self.enc_fc_size))
                self.transition_params.extend(self.bn9_list[i].parameters())
        else:
            
            self.fc_trans_3 = nn.Linear(self.transition_size, self.transition_size)
            self.transition_params.extend(self.fc_trans_3.parameters())
            self.bn9_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn9_list.append(nn.BatchNorm1d(self.transition_size))
                self.transition_params.extend(self.bn9_list[i].parameters())
                
            self.fc_trans_4 = nn.Linear(self.transition_size, self.transition_size)
            self.transition_params.extend(self.fc_trans_4.parameters())
            self.bn10_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn10_list.append(nn.BatchNorm1d(self.transition_size))
                self.transition_params.extend(self.bn10_list[i].parameters())
                
                
            self.fc_trans_5 = nn.Linear(self.transition_size, self.enc_fc_size)
            self.transition_params.extend(self.fc_trans_5.parameters())
            self.bn11_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn11_list.append(nn.BatchNorm1d(self.enc_fc_size))
                self.transition_params.extend(self.bn11_list[i].parameters())
            
               
        ### decoder #####
        self.decoder_params = []
        
        self.fc_z_x_1 = nn.Linear(args.nl, (self.init_ch*4)*(input_shape[1]/8)*(input_shape[2]/8))
        self.decoder_params.extend(self.fc_z_x_1.parameters())
        self.bn12_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn12_list.append(nn.BatchNorm1d((self.init_ch*4)*(input_shape[1]/8)*(input_shape[2]/8)))
            self.decoder_params.extend(self.bn12_list[i].parameters())
            
            
        self.conv_z_x_2 = nn.ConvTranspose2d(self.init_ch*4, self.init_ch*2, kernel_size=4, stride= 2, padding=1)
        self.decoder_params.extend(self.conv_z_x_2.parameters())
        self.bn13_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn13_list.append(nn.BatchNorm2d(self.init_ch*2))
            self.decoder_params.extend(self.bn13_list[i].parameters())
            
            
        self.conv_z_x_3 = nn.ConvTranspose2d(self.init_ch*2, self.init_ch, kernel_size=4, stride= 2, padding=1)
        self.decoder_params.extend(self.conv_z_x_3.parameters())
        self.bn14_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn14_list.append(nn.BatchNorm2d(self.init_ch))
            self.decoder_params.extend(self.bn14_list[i].parameters())
            
            
        self.conv_z_x_4 = nn.ConvTranspose2d(self.init_ch, 3, kernel_size=4, stride= 2, padding=1)
        self.decoder_params.extend(self.conv_z_x_4.parameters())
        self.bn15_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn15_list.append(nn.BatchNorm2d(3))
            self.decoder_params.extend(self.bn15_list[i].parameters())
            
           
            
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
    
    def encode(self, x, step):
        
        c1 = self.act(self.bn1_list[step](self.conv_x_z_1(x)))
        #print c1
        c2 = self.act(self.bn2_list[step](self.conv_x_z_2(c1)))
        #print c2
        c3 = self.act(self.bn3_list[step](self.conv_x_z_3(c2)))
        #print c3
        c3 = c3.view(-1, self.flat_shape)
        h1 = self.act(self.bn4_list[step](self.fc_layer_1(c3)))
        #print h1
        mu = self.bn5_list[step](self.fc_z_mu(h1))
        sigma = self.bn6_list[step](self.fc_z_sigma(h1))
        return mu, sigma

        
        
    def reparameterize(self, mu, sigma):
        std = sigma.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    
       
    def transition (self, z, temperature, step):
        #print ('z', np.isnan(z.data.cpu().numpy()).any())
        #    print z.requires_grad
        h = self.act(self.bn7_list[step](self.fc_trans_1(z)))
        #print h1
        h = self.act(self.bn8_list[step](self.fc_trans_2(h)))
        #print h2
        h = self.act(self.bn9_list[step](self.fc_trans_3(h)))
        if self.args.transition_steps>3:
            h = self.act(self.bn10_list[step](self.fc_trans_4(h)))
            h = self.act(self.bn11_list[step](self.fc_trans_5(h)))
            
        #print h3
        h = torch.clamp(h, min=0, max=5)
        #print h3
        mu = self.bn5_list[step](self.fc_z_mu(h))  #### use h3 for three layers in the transition operator
        #print mu
        sigma = self.bn6_list[step](self.fc_z_sigma(h))
        #print sigma
        #print ('mu', np.isnan(mu.data.cpu().numpy()).any())
        #print ('sigma', np.isnan(sigma.data.cpu().numpy()).any())
        eps = Variable(mu.data.new(mu.size()).normal_())
        
        #print ('eps', np.isnan(eps.data.cpu().numpy()).any())
        #z_new = mu + T.sqrt(args.sigma * temperature) * T.exp(0.5 * sigma) * eps
        #z_new = (z_new - T.mean(z_new, axis=0, keepdims=True)) / (0.001 + T.std(z_new, axis=0, keepdims=True))
        
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
        d = self.act(self.bn12_list[step](self.fc_z_x_1(z_new)))
        #print d0
        d = d.view(-1, (self.init_ch*4),(self.input_shape[1]/8), (self.input_shape[2]/8))
        d = self.act(self.bn13_list[step](self.conv_z_x_2(d)))
        #print d1.data.shape
        d = self.act(self.bn14_list[step](self.conv_z_x_3(d)))
        #print self.conv_z_x_3(d1)
        #print d2.data.shape
        d = self.sigmoid(self.bn15_list[step](self.conv_z_x_4(d)))
        #print self.conv_z_x_4(d2)
        #print d3.data.shape
        shape = d.data.shape
        p =  d.view(-1, shape[1]*shape[2]*shape[3])
        
        eps = 1e-4
        p = torch.clamp(p, min= eps, max=1.0 - eps)
        #x_loss =  -T.nnet.binary_crossentropy(p, x).sum(axis=1)
        return p
    
    def sample(self, z, temperature,step):
        d = self.act(self.bn12_list[step](self.fc_z_x_1(z)))
        d = d.view(-1, (self.init_ch*4), (self.input_shape[1]/8), (self.input_shape[2]/8))
        d = self.act(self.bn13_list[step](self.conv_z_x_2(d)))
        d = self.act(self.bn14_list[step](self.conv_z_x_3(d)))
        d = self.sigmoid(self.bn15_list[step](self.conv_z_x_4(d)))
        shape = d.data.shape
        x_new =  d.view(-1, shape[1]*shape[2]*shape[3])
    
        z_new, log_p_reverse, sigma, h2 = self.transition( z , temperature, step)
        x_tilde = self.decode(z_new,step)
        
        return x_tilde, x_new, z_new 
    
"""    
    
    
    
class Net_cifar(nn.Module):
    def __init__(self, args, input_shape=(3,32,32)):
        super(Net_cifar, self).__init__()
        print 'new net'
        self.args = args
        self.init_ch = args.init_ch
        self.input_shape = input_shape
        self.enc_fc_size = args.enc_fc_size
        self.transition_size = args.transition_size
        self.kernel_size = args.kernel_size # TODO : add in args
        self.stride = args.stride## TODO : add in args
        padsize = 2
        self.imgSize = input_shape
        
        if self.args.activation == 'relu':
            self.act = nn.ReLU()
        elif self.args.activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        ### Encoder ######
        self.encoder_params= []   
         
        self.conv_x_z_1 = nn.Conv2d(self.imgSize[0], self.init_ch, kernel_size=self.kernel_size, stride=self.stride)
        self.encoder_params.extend(self.conv_x_z_1.parameters())
        self.bn1_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn1_list.append(nn.BatchNorm2d(self.init_ch))
            self.encoder_params.extend(self.bn1_list[i].parameters())
        
        self.conv_x_z_2 = nn.Conv2d(self.init_ch, self.init_ch*2, kernel_size = self.kernel_size, stride= self.stride)
        self.encoder_params.extend(self.conv_x_z_2.parameters())
        self.bn2_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn2_list.append(nn.BatchNorm2d(self.init_ch*2))
            self.encoder_params.extend(self.bn2_list[i].parameters())
        
        self.conv_x_z_3 = nn.Conv2d(self.init_ch*2, self.init_ch*4, kernel_size=self.kernel_size, stride = self.stride)
        self.encoder_params.extend(self.conv_x_z_3.parameters())
        self.bn3_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn3_list.append(nn.BatchNorm2d(self.init_ch*4))
            self.encoder_params.extend(self.bn3_list[i].parameters())
            
        self.flat_shape, self.last_encoder_shape = self.get_flat_shape(input_shape) 
        #self.fc_layer_1 = nn.Linear(self.flat_shape, self.enc_fc_size)
        #self.encoder_params.extend(self.fc_layer_1.parameters())
        #self.bn4_list=nn.ModuleList()
        #for i in xrange(args.meta_steps):
        #    self.bn4_list.append(nn.BatchNorm1d(self.enc_fc_size))
        #    self.encoder_params.extend(self.bn4_list[i].parameters())
            
        self.fc_z_mu = nn.Linear(self.flat_shape, args.nl)
        self.encoder_params.extend(self.fc_z_mu.parameters())
        self.bn5_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn5_list.append(nn.BatchNorm1d(args.nl))
            self.encoder_params.extend(self.bn5_list[i].parameters())
        
        self.fc_z_sigma = nn.Linear(self.flat_shape, args.nl)
        self.encoder_params.extend(self.fc_z_sigma.parameters())
        self.bn6_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn6_list.append(nn.BatchNorm1d(args.nl))
            self.encoder_params.extend(self.bn6_list[i].parameters())
        
       
        ###### transition operator ########
        self.transition_params = []
        
        self.fc_trans_1 = nn.Linear(args.nl, self.transition_size)
        self.transition_params.extend(self.fc_trans_1.parameters())
        self.bn7_list=nn.ModuleList()
        #print args.meta_steps
        for i in xrange(args.meta_steps):
            self.bn7_list.append(nn.BatchNorm1d(self.transition_size))
            self.transition_params.extend(self.bn7_list[i].parameters())
            #print _
       
        self.fc_trans_2 = nn.Linear(self.transition_size, self.transition_size)
        self.transition_params.extend(self.fc_trans_2.parameters())
        self.bn8_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn8_list.append(nn.BatchNorm1d(self.transition_size))
            self.transition_params.extend(self.bn8_list[i].parameters())
        
        if self.args.transition_steps==3:
            self.fc_trans_3 = nn.Linear(self.transition_size, self.flat_shape)
            self.transition_params.extend(self.fc_trans_3.parameters())
            self.bn9_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn9_list.append(nn.BatchNorm1d(self.flat_shape))
                self.transition_params.extend(self.bn9_list[i].parameters())
        else:
            
            self.fc_trans_3 = nn.Linear(self.transition_size, self.transition_size)
            self.transition_params.extend(self.fc_trans_3.parameters())
            self.bn9_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn9_list.append(nn.BatchNorm1d(self.transition_size))
                self.transition_params.extend(self.bn9_list[i].parameters())
                
            self.fc_trans_4 = nn.Linear(self.transition_size, self.transition_size)
            self.transition_params.extend(self.fc_trans_4.parameters())
            self.bn10_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn10_list.append(nn.BatchNorm1d(self.transition_size))
                self.transition_params.extend(self.bn10_list[i].parameters())
                
                
            self.fc_trans_5 = nn.Linear(self.transition_size, self.flat_shape)
            self.transition_params.extend(self.fc_trans_5.parameters())
            self.bn11_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn11_list.append(nn.BatchNorm1d(self.flat_shape))
                self.transition_params.extend(self.bn11_list[i].parameters())
            
               
        ### decoder #####
        self.decoder_params = []
        
        self.fc_z_x_1 = nn.Linear(args.nl, self.flat_shape)
        self.decoder_params.extend(self.fc_z_x_1.parameters())
        self.bn12_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn12_list.append(nn.BatchNorm1d(self.flat_shape))
            self.decoder_params.extend(self.bn12_list[i].parameters())
            
            
        self.conv_z_x_1 = nn.ConvTranspose2d(self.init_ch*4, self.init_ch*2, kernel_size=self.kernel_size, stride= self.stride)
        self.decoder_params.extend(self.conv_z_x_1.parameters())
        self.bn13_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn13_list.append(nn.BatchNorm2d(self.init_ch*2))
            self.decoder_params.extend(self.bn13_list[i].parameters())
            
            
        self.conv_z_x_2 = nn.ConvTranspose2d(self.init_ch*2, self.init_ch, kernel_size = self.kernel_size, stride = self.stride)
        self.decoder_params.extend(self.conv_z_x_2.parameters())
        self.bn14_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn14_list.append(nn.BatchNorm2d(self.init_ch))
            self.decoder_params.extend(self.bn14_list[i].parameters())
            
            
        self.conv_z_x_3 = nn.ConvTranspose2d(self.init_ch, self.imgSize[0], kernel_size= self.kernel_size, stride= self.stride)
        self.decoder_params.extend(self.conv_z_x_3.parameters())
        self.bn15_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn15_list.append(nn.BatchNorm2d(self.imgSize[0]))
            self.decoder_params.extend(self.bn15_list[i].parameters())
            
        
        #self.conv_z_x_4 = nn.Conv2d(self.init_ch, self.imgSize[0], kernel_size= 1, stride= 1)
        #self.decoder_params.extend(self.conv_z_x_4.parameters())
        #self.bn16_list=nn.ModuleList()
        #for i in xrange(args.meta_steps):
        #    self.bn16_list.append(nn.BatchNorm2d(self.imgSize[0]))
        #    self.decoder_params.extend(self.bn16_list[i].parameters())
           
            
    def get_flat_shape(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.conv_x_z_1(dummy)
        dummy = self.conv_x_z_2(dummy)
        dummy = self.conv_x_z_3(dummy)
        
        return dummy.data.view(1, -1).size(1), dummy.data.shape
    
    
    def encode(self, x, step):
        
        #h = self.act(self.bn1_list[step](self.conv_x_z_1(x)))
        h = self.act(self.conv_x_z_1(x))
        
        #print h.shape
        #h = self.act(self.bn2_list[step](self.conv_x_z_2(h)))
        h = self.act(self.conv_x_z_2(h))
        
        #print h.shape
        #h = self.act(self.bn3_list[step](self.conv_x_z_3(h)))
        h = self.act(self.conv_x_z_3(h))
        
        
        #print h.shape
        h = h.view(-1, self.flat_shape)
        #h = self.act(self.bn4_list[step](self.fc_layer_1(h)))
        #print h.shape
        mu = self.fc_z_mu(h)#mu = self.bn5_list[step](self.fc_z_mu(h))
        sigma = self.fc_z_sigma(h)#sigma = self.bn6_list[step](self.fc_z_sigma(h))
        return mu, sigma

        
        
    def reparameterize(self, mu, sigma):
        std = sigma.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    
       
    def transition (self, z, temperature, step):
        #print ('z', np.isnan(z.data.cpu().numpy()).any())
        #    print z.requires_grad
        h = self.act(self.bn7_list[step](self.fc_trans_1(z)))
        #print h.shape
        h = self.act(self.bn8_list[step](self.fc_trans_2(h)))
        #print h.shape
        h = self.act(self.bn9_list[step](self.fc_trans_3(h)))
        if self.args.transition_steps>3:
            h = self.act(self.bn10_list[step](self.fc_trans_4(h)))
            h = self.act(self.bn11_list[step](self.fc_trans_5(h)))
            
        #print h.shape
        #h = torch.clamp(h, min=0, max=5)
        #print h3
        mu =  self.fc_z_mu(h)#self.bn5_list[step](self.fc_z_mu(h))  #### use h3 for three layers in the transition operator
        #print mu
        sigma = self.fc_z_sigma(h)#self.bn6_list[step](self.fc_z_sigma(h))
        #print sigma
        #print ('mu', np.isnan(mu.data.cpu().numpy()).any())
        #print ('sigma', np.isnan(sigma.data.cpu().numpy()).any())
        eps = Variable(mu.data.new(mu.size()).normal_())
        
        #print ('eps', np.isnan(eps.data.cpu().numpy()).any())
        #z_new = mu + T.sqrt(args.sigma * temperature) * T.exp(0.5 * sigma) * eps
        #z_new = (z_new - T.mean(z_new, axis=0, keepdims=True)) / (0.001 + T.std(z_new, axis=0, keepdims=True))
        
        if self.args.cuda:
            sigma_ = Variable(torch.sqrt(torch.FloatTensor(1).fill_(self.args.sigma * temperature)).cuda())
            #print ('sigma_', np.isnan(sigma_.data.cpu().numpy()).any())
        
        else:
            sigma_ = Variable(torch.sqrt(torch.FloatTensor(1).fill_(self.args.sigma * temperature)))
        
        z_new = eps.mul(sigma.mul(0.5).exp_()).mul(sigma_).add_(mu)
        #print ('z_new', np.isnan(z_new.data.cpu().numpy()).any())
        ##z_new = (z_new - z_new.mean(0))/(0.001+ z_new.std(0))##
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
        ##z_new = torch.clamp(z_new, min=-4, max=4)##
        #print z_new 
        return z_new, log_p_reverse, mu, sigma
        
     
    def decode (self, z_new, step):
        #print z_new
        #d = self.act(self.bn12_list[step](self.fc_z_x_1(z_new)))
        d = self.act(self.fc_z_x_1(z_new))
        
        #print (d.shape)
        d = d.view(-1, self.last_encoder_shape[1],self.last_encoder_shape[2], self.last_encoder_shape[3])
        
        #print (d.shape)
        #d = self.act(self.bn13_list[step](self.conv_z_x_1(d)))
        d = self.act(self.conv_z_x_1(d))
        
        #print (d.shape)
        #d = self.act(self.bn14_list[step](self.conv_z_x_2(d)))
        d = self.act(self.conv_z_x_2(d))
        
        #print (d.shape)
        #d = self.sigmoid(self.bn15_list[step](self.conv_z_x_3(d)))
        d = self.sigmoid(self.conv_z_x_3(d))
        
        #print (d.shape)
        #d = self.sigmoid(self.bn16_list[step](self.conv_z_x_4(d)))
        #print (d.shape)
        shape = d.data.shape
        p =  d.view(-1, shape[1]*shape[2]*shape[3])
        eps = 1e-4
        p = torch.clamp(p, min= eps, max=1.0 - eps)
        #x_loss =  -T.nnet.binary_crossentropy(p, x).sum(axis=1)
        return p
    
    def sample(self, z, temperature,step):
        
        #d = self.act(self.bn12_list[step](self.fc_z_x_1(z)))
        d = self.act(self.fc_z_x_1(z))
        
        d = d.view(-1, self.last_encoder_shape[1],self.last_encoder_shape[2], self.last_encoder_shape[3])
        
        #d = self.act(self.bn13_list[step](self.conv_z_x_1(d)))
        d = self.act(self.conv_z_x_1(d))
        
        #d = self.act(self.bn14_list[step](self.conv_z_x_2(d)))
        d = self.act(self.conv_z_x_2(d))
        
        #d = self.sigmoid(self.bn15_list[step](self.conv_z_x_3(d)))
        d = self.sigmoid(self.conv_z_x_3(d))
        
        
        shape = d.data.shape
        x_new =  d.view(-1, shape[1]*shape[2]*shape[3])
    
        z_new, log_p_reverse, sigma, h2 = self.transition( z , temperature, step)
        x_tilde = self.decode(z_new,step)
        
        return x_tilde, x_new, z_new 
    
    
    
class Net_svhn(nn.Module):
    ## code replicating anirudh's theano code
    def __init__(self, args, input_shape=(3,32,32)):
        super(Net_svhn, self).__init__()
        print ('damn')
        self.args = args
        self.init_ch = args.init_ch
        self.input_shape = input_shape
        self.enc_fc_size = args.enc_fc_size
        self.transition_size = args.transition_size
        self.stride = args.stride
        self.kernel_size = args.kernel_size
        padsize = 1
        print ('pad')
        
        if self.args.activation == 'relu':
            self.act = nn.ReLU()
        elif self.args.activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        ### Encoder ######
        self.encoder_params= []   
         
        self.conv_x_z_1 = nn.Conv2d(input_shape[0], self.init_ch, kernel_size=self.kernel_size, stride=self.stride)#, padding= )
        self.encoder_params.extend(self.conv_x_z_1.parameters())
        self.bn1_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn1_list.append(nn.BatchNorm2d(self.init_ch))
            self.encoder_params.extend(self.bn1_list[i].parameters())
        
        self.conv_x_z_2 = nn.Conv2d(self.init_ch, self.init_ch*2, kernel_size=self.kernel_size, stride=self.stride)#,  padding= padsize)
        self.encoder_params.extend(self.conv_x_z_2.parameters())
        self.bn2_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn2_list.append(nn.BatchNorm2d(self.init_ch*2))
            self.encoder_params.extend(self.bn2_list[i].parameters())
        
        self.conv_x_z_3 = nn.Conv2d(self.init_ch*2, self.init_ch*4, kernel_size=self.kernel_size, stride=self.stride)#, padding= padsize)
        self.encoder_params.extend(self.conv_x_z_3.parameters())
        self.bn3_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn3_list.append(nn.BatchNorm2d(self.init_ch*4))
            self.encoder_params.extend(self.bn3_list[i].parameters())
           
        self.flat_shape, self.last_encoder_shape = self.get_flat_shape(input_shape)
        
        self.fc_layer_1 = nn.Linear(self.flat_shape, self.enc_fc_size)
        self.encoder_params.extend(self.fc_layer_1.parameters())
        self.bn4_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn4_list.append(nn.BatchNorm1d(self.enc_fc_size))
            self.encoder_params.extend(self.bn4_list[i].parameters())
            
        self.fc_z_mu = nn.Linear(self.enc_fc_size, args.nl)
        self.encoder_params.extend(self.fc_z_mu.parameters())
        self.bn5_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn5_list.append(nn.BatchNorm1d(args.nl))
            self.encoder_params.extend(self.bn5_list[i].parameters())
        
        self.fc_z_sigma = nn.Linear(self.enc_fc_size, args.nl)
        self.encoder_params.extend(self.fc_z_sigma.parameters())
        self.bn6_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn6_list.append(nn.BatchNorm1d(args.nl))
            self.encoder_params.extend(self.bn6_list[i].parameters())
        
       
        ###### transition operator ########
        self.transition_params = []
        
        self.fc_trans_1 = nn.Linear(args.nl, self.transition_size)
        self.transition_params.extend(self.fc_trans_1.parameters())
        self.bn7_list=nn.ModuleList()
        #print args.meta_steps
        for i in xrange(args.meta_steps):
            self.bn7_list.append(nn.BatchNorm1d(self.transition_size))
            self.transition_params.extend(self.bn7_list[i].parameters())
            #print _
       
        self.fc_trans_2 = nn.Linear(self.transition_size, self.transition_size)
        self.transition_params.extend(self.fc_trans_2.parameters())
        self.bn8_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn8_list.append(nn.BatchNorm1d(self.transition_size))
            self.transition_params.extend(self.bn8_list[i].parameters())
        
        if self.args.transition_steps==3:
            self.fc_trans_3 = nn.Linear(self.transition_size, self.enc_fc_size)
            self.transition_params.extend(self.fc_trans_3.parameters())
            self.bn9_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn9_list.append(nn.BatchNorm1d(self.enc_fc_size))
                self.transition_params.extend(self.bn9_list[i].parameters())
        else:
            
            self.fc_trans_3 = nn.Linear(self.transition_size, self.transition_size)
            self.transition_params.extend(self.fc_trans_3.parameters())
            self.bn9_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn9_list.append(nn.BatchNorm1d(self.transition_size))
                self.transition_params.extend(self.bn9_list[i].parameters())
                
            self.fc_trans_4 = nn.Linear(self.transition_size, self.transition_size)
            self.transition_params.extend(self.fc_trans_4.parameters())
            self.bn10_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn10_list.append(nn.BatchNorm1d(self.transition_size))
                self.transition_params.extend(self.bn10_list[i].parameters())
                
                
            self.fc_trans_5 = nn.Linear(self.transition_size, self.enc_fc_size)
            self.transition_params.extend(self.fc_trans_5.parameters())
            self.bn11_list=nn.ModuleList()
            for i in xrange(args.meta_steps):
                self.bn11_list.append(nn.BatchNorm1d(self.enc_fc_size))
                self.transition_params.extend(self.bn11_list[i].parameters())
            
               
        ### decoder #####
        self.decoder_params = []
        
        self.fc_z_x_1 = nn.Linear(args.nl, self.flat_shape)
        self.decoder_params.extend(self.fc_z_x_1.parameters())
        self.bn12_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn12_list.append(nn.BatchNorm1d(self.flat_shape))
            self.decoder_params.extend(self.bn12_list[i].parameters())
            
            
        self.conv_z_x_2 = nn.ConvTranspose2d(self.init_ch*4, self.init_ch*2, kernel_size=self.kernel_size, stride= self.stride)
        self.decoder_params.extend(self.conv_z_x_2.parameters())
        self.bn13_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn13_list.append(nn.BatchNorm2d(self.init_ch*2))
            self.decoder_params.extend(self.bn13_list[i].parameters())
            
            
        self.conv_z_x_3 = nn.ConvTranspose2d(self.init_ch*2, self.init_ch, kernel_size= self.kernel_size, stride= self.stride)
        self.decoder_params.extend(self.conv_z_x_3.parameters())
        self.bn14_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn14_list.append(nn.BatchNorm2d(self.init_ch))
            self.decoder_params.extend(self.bn14_list[i].parameters())
            
            
        self.conv_z_x_4 = nn.ConvTranspose2d(self.init_ch, self.input_shape[0], kernel_size=self.kernel_size, stride= self.stride)
        self.decoder_params.extend(self.conv_z_x_4.parameters())
        self.bn15_list=nn.ModuleList()
        for i in xrange(args.meta_steps):
            self.bn15_list.append(nn.BatchNorm2d(self.input_shape[0]))
            self.decoder_params.extend(self.bn15_list[i].parameters())
            
           
            
    def get_flat_shape(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.conv_x_z_1(dummy)
        dummy = self.conv_x_z_2(dummy)
        dummy = self.conv_x_z_3(dummy)
        
        return dummy.data.view(1, -1).size(1), dummy.data.shape
    
    
       
    def encode(self, x, step):
        c = self.act(self.bn1_list[step](self.conv_x_z_1(x)))
        #print (c.shape)
        c = self.act(self.bn2_list[step](self.conv_x_z_2(c)))
        #print (c.shape)
        c = self.act(self.bn3_list[step](self.conv_x_z_3(c)))
        #print (c.shape)
        c = c.view(-1, self.flat_shape)
        h = self.act(self.bn4_list[step](self.fc_layer_1(c)))
        #print (h.shape)
        
        mu = self.fc_z_mu(h)#mu = self.bn5_list[step](self.fc_z_mu(h1))
        sigma = self.fc_z_sigma(h)#sigma = self.bn6_list[step](self.fc_z_sigma(h1))
        return mu, sigma

        
        
    def reparameterize(self, mu, sigma):
        std = sigma.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    
       
    def transition (self, z, temperature, step):
        #print ('z', np.isnan(z.data.cpu().numpy()).any())
        #    print z.requires_grad
        h = self.act(self.bn7_list[step](self.fc_trans_1(z)))
        #print (h.shape)
        h = self.act(self.bn8_list[step](self.fc_trans_2(h)))
        #print (h.shape)
        h = self.act(self.bn9_list[step](self.fc_trans_3(h)))
        #print (h.shape)
        
        if self.args.transition_steps>3:
            h = self.act(self.bn10_list[step](self.fc_trans_4(h)))
            h = self.act(self.bn11_list[step](self.fc_trans_5(h)))
            
        #print h3
        #h = torch.clamp(h, min=0, max=5)
        #print (h.shape)
        mu = self.fc_z_mu(h)#mu = self.bn5_list[step](self.fc_z_mu(h))  #### use h3 for three layers in the transition operator
        #print mu
        sigma = self.fc_z_sigma(h)#sigma = self.bn6_list[step](self.fc_z_sigma(h))
        #print sigma
        #print ('mu', np.isnan(mu.data.cpu().numpy()).any())
        #print ('sigma', np.isnan(sigma.data.cpu().numpy()).any())
        eps = Variable(mu.data.new(mu.size()).normal_())
        
        #print ('eps', np.isnan(eps.data.cpu().numpy()).any())
        #z_new = mu + T.sqrt(args.sigma * temperature) * T.exp(0.5 * sigma) * eps
        #z_new = (z_new - T.mean(z_new, axis=0, keepdims=True)) / (0.001 + T.std(z_new, axis=0, keepdims=True))
        
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
        d = self.act(self.bn12_list[step](self.fc_z_x_1(z_new)))
        #print (d.shape)
        d = d.view(-1, self.last_encoder_shape[1],self.last_encoder_shape[2], self.last_encoder_shape[3])
        #print (d.shape)
        d = self.act(self.bn13_list[step](self.conv_z_x_2(d)))
        #print (d.shape)
        d = self.act(self.bn14_list[step](self.conv_z_x_3(d)))
        #print self.conv_z_x_3(d1)
        #print (d.shape)
        d = self.sigmoid(self.bn15_list[step](self.conv_z_x_4(d)))
        #print (d.shape)
        #print self.conv_z_x_4(d2)
        #print d3.data.shape
        shape = d.data.shape
        p =  d.view(-1, shape[1]*shape[2]*shape[3])
        
        eps = 1e-4
        p = torch.clamp(p, min= eps, max=1.0 - eps)
        #x_loss =  -T.nnet.binary_crossentropy(p, x).sum(axis=1)
        return p
    
    def sample(self, z, temperature,step):
        d = self.act(self.bn12_list[step](self.fc_z_x_1(z)))
        d = d.view(-1, self.last_encoder_shape[1],self.last_encoder_shape[2], self.last_encoder_shape[3])
        d = self.act(self.bn13_list[step](self.conv_z_x_2(d)))
        d = self.act(self.bn14_list[step](self.conv_z_x_3(d)))
        d = self.sigmoid(self.bn15_list[step](self.conv_z_x_4(d)))
        shape = d.data.shape
        x_new =  d.view(-1, shape[1]*shape[2]*shape[3])
    
        z_new, log_p_reverse, sigma, h2 = self.transition( z , temperature, step)
        x_tilde = self.decode(z_new,step)
        
        return x_tilde, x_new, z_new 
