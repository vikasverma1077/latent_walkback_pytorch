import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import *

class VAE_old(nn.Module):
    def __init__(self, args, imgSize=(3,64,64)):
        super(VAE_old, self).__init__()
        self.init_ch = args.init_ch
        self.args = args
        self.imgSize = imgSize
        ## encoder###
        self.conv1 = nn.Conv2d(imgSize[0], self.init_ch, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(self.init_ch)
        self.conv2 = nn.Conv2d(self.init_ch, self.init_ch, 3, stride=1)
        self.bn2 = nn.BatchNorm2d(self.init_ch)
        self.conv3= nn.Conv2d(self.init_ch, self.init_ch*2, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(self.init_ch*2)
        #self.conv4 = nn.Conv2d(64, 64, 4, stride=2)
        #self.bn4 = nn.BatchNorm2d(64)
        self.flat_shape = self.get_flat_shape(imgSize)
        
        self.fc1 = nn.Linear(self.flat_shape, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc21 = nn.Linear(128, self.args.nl)
        #self.bn5_1 = nn.BatchNorm1d(self.args.nl)
        self.fc22 = nn.Linear(128, self.args.nl)
        #self.bn5_2 = nn.BatchNorm1d(self.args.nl)
        ### decoder###
        self.fc3 = nn.Linear(self.args.nl, 128)
        self.bn6 = nn.BatchNorm1d(128)
        #self.fc4 = nn.Linear(128, (imgSize[1]/8)*(imgSize[1]/8)*64)
        #self.bn7 = nn.BatchNorm1d((imgSize[1]/8)*(imgSize[1]/8)*64)
        
        self.fc4 = nn.Linear(128, 26*26*self.init_ch*2)
        self.bn7 = nn.BatchNorm1d(26*26*self.init_ch*2)
        
        
        self.conv_z_1 = nn.ConvTranspose2d(self.init_ch*2, self.init_ch, kernel_size=3, stride= 1)#, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.conv_z_2 = nn.ConvTranspose2d(self.init_ch, self.init_ch, kernel_size=3, stride= 1)#, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.conv_z_3 = nn.ConvTranspose2d(self.init_ch, imgSize[0], kernel_size=3, stride= 1)#, padding=1)
        self.bn10 = nn.BatchNorm2d(imgSize[0])
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.imgSize = imgSize
    
    def get_flat_shape(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.conv1(dummy)
        dummy = self.conv2(dummy)
        dummy = self.conv3(dummy)
        #dummy = self.conv4(dummy)
        return dummy.data.view(1, -1).size(1)
    
    def encode(self, x):
        #x = x.view(-1,1,64,64)
        h = self.relu(self.bn1(self.conv1(x)))
        #print h.shape
        h = self.relu(self.bn2(self.conv2(h)))
        #print h.shape
        h = self.relu(self.bn3(self.conv3(h)))
        #print h.shape
        #h = self.relu(self.bn4(self.conv4(h)))
        #print (h.shape)
        h = h.view(-1, self.flat_shape)
        #print (h.shape)
        h = self.relu(self.bn5(self.fc1(h)))
        return  self.fc21(h), self.fc22(h)  #self.bn5_1(self.fc21(h)), self.bn5_2(self.fc22(h))

    def reparameterize(self, mu, logvar):
        #if self.training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
        #else:
        #return mu

    def decode(self, z):
        #print (z.shape)
        h = self.relu(self.bn6(self.fc3(z)))
        #print h.shape
        h = self.relu(self.bn7(self.fc4(h)))
        #print h.shape
        #h = h.view(-1, 64, (self.imgSize[1]/8),(self.imgSize[1]/8))
        h = h.view(-1, self.init_ch*2, 26, 26)
        #print h.shape
        h = self.relu(self.bn8(self.conv_z_1(h)))
        #print h.shape
        h = self.relu(self.bn9(self.conv_z_2(h)))
        #print h.shape
        #h9 = self.relu(self.conv_z_3(h8))
        return self.sigmoid(self.bn10(self.conv_z_3(h)))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, *(self.imgSize)))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
    



class VAE(nn.Module):
    def __init__(self, args, imgSize=(3,64,64)):
        super(VAE, self).__init__()
        self.init_ch = args.init_ch
        print ('init_ch=', self.init_ch)
        self.kernel_size =2
        self.stride = 2
        self.args = args
        self.imgSize = imgSize
        ## encoder###
        self.conv1 = nn.Conv2d(imgSize[0], self.init_ch, self.kernel_size, stride = self.stride)
        kaiming_uniform(self.conv1.weight.data)
        self.bn1 = nn.BatchNorm2d(self.init_ch)
        self.conv2 = nn.Conv2d(self.init_ch, self.init_ch*2, self.kernel_size, stride = self.stride)
        kaiming_uniform(self.conv2.weight.data)
        self.bn2 = nn.BatchNorm2d(self.init_ch*2)
        self.conv3= nn.Conv2d(self.init_ch*2, self.init_ch*4, self.kernel_size, stride = self.stride)
        kaiming_uniform(self.conv3.weight.data)
        self.bn3 = nn.BatchNorm2d(self.init_ch*4)
        #self.conv4 = nn.Conv2d(64, 64, 4, stride=2)
        #self.bn4 = nn.BatchNorm2d(64)
        self.flat_shape, self.last_encoder_shape = self.get_flat_shape(imgSize)
        
        #print self.flat_shape, self.last_encoder_shape
        #self.fc1 = nn.Linear(self.flat_shape, 128)
        #self.bn5 = nn.BatchNorm1d(128)
        self.fc21 = nn.Linear(self.flat_shape, self.args.nl)
        kaiming_uniform(self.fc21.weight.data)
        #self.bn5_1 = nn.BatchNorm1d(self.args.nl)
        self.fc22 = nn.Linear(self.flat_shape, self.args.nl)
        kaiming_uniform(self.fc22.weight.data)
        #self.bn5_2 = nn.BatchNorm1d(self.args.nl)
        
        ### decoder###
        self.fc3 = nn.Linear(self.args.nl, self.flat_shape)
        kaiming_uniform(self.fc3.weight.data)
              
        self.conv_z_1 = nn.ConvTranspose2d(self.init_ch*4, self.init_ch*4, kernel_size= self.kernel_size, stride= self.stride)#, padding=1)
        kaiming_uniform(self.conv_z_1.weight.data)
        self.bn8 = nn.BatchNorm2d(self.init_ch*4)
        
        self.conv_z_2 = nn.ConvTranspose2d(self.init_ch*4, self.init_ch*2, kernel_size= self.kernel_size, stride= self.stride)#, padding=1)
        kaiming_uniform(self.conv_z_2.weight.data)
        self.bn9 = nn.BatchNorm2d(self.init_ch*2)
        
        self.conv_z_3 = nn.ConvTranspose2d(self.init_ch*2, self.init_ch, kernel_size= self.kernel_size, stride= self.stride)#, padding=1)
        kaiming_uniform(self.conv_z_3.weight.data)
        self.bn10 = nn.BatchNorm2d(self.init_ch)
        
        self.conv_z_4 = nn.Conv2d(self.init_ch, self.imgSize[0], kernel_size= 1, stride= 1)#, padding=1)
        kaiming_uniform(self.conv_z_4.weight.data)
        self.bn11 = nn.BatchNorm2d(self.imgSize[0])
        
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.imgSize = imgSize
    
    def get_flat_shape(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.conv1(dummy)
        dummy = self.conv2(dummy)
        dummy = self.conv3(dummy)
        #dummy = self.conv4(dummy)
        return dummy.data.view(1, -1).size(1), dummy.data.shape
    
    def encode(self, x):
        #x = x.view(-1,1,64,64)
        h = self.relu(self.bn1(self.conv1(x)))
        #print h.shape
        h = self.relu(self.bn2(self.conv2(h)))
        #print h.shape
        h = self.relu(self.bn3(self.conv3(h)))
        #print h.shape
        #h = self.relu(self.bn4(self.conv4(h)))
        #print (h.shape)
        h = h.view(-1, self.flat_shape)
        #print (h.shape)
        #h = self.relu(self.bn5(self.fc1(h)))
        return  self.fc21(h), self.fc22(h)  #self.bn5_1(self.fc21(h)), self.bn5_2(self.fc22(h))

    def reparameterize(self, mu, logvar):
        #if self.training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
        #else:
        #return mu

    def decode(self, z):
        #print (z.shape)
        h = self.relu((self.fc3(z)))
        #print h.shape
        h = h.view(-1, self.last_encoder_shape[1], self.last_encoder_shape[2], self.last_encoder_shape[3])
        #print h.shape
        h = self.relu(self.bn8(self.conv_z_1(h)))
        #print h.shape
        h = self.relu(self.bn9(self.conv_z_2(h)))
        #print h.shape
        h = self.relu(self.bn10(self.conv_z_3(h)))
        #print h.shape
        #h9 = self.relu(self.conv_z_3(h8))
        x = self.sigmoid(self.bn11(self.conv_z_4(h)))
        #print x.shape
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, *(self.imgSize)))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

