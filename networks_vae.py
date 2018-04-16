import torch
import torch.nn as nn
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, args, imgSize=(1,64,64)):
        super(VAE, self).__init__()
        
        self.args = args
        self.imgSize = imgSize
        ## encoder###
        self.conv1 = nn.Conv2d(imgSize[0], 32, 4, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3= nn.Conv2d(32, 64, 4, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        #self.conv4 = nn.Conv2d(64, 64, 4, stride=2)
        #self.bn4 = nn.BatchNorm2d(64)
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
        self.fc4 = nn.Linear(128, (imgSize[1]/8)*(imgSize[1]/8)*64)
        self.bn7 = nn.BatchNorm1d((imgSize[1]/8)*(imgSize[1]/8)*64)
        
        self.conv_z_1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride= 2, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.conv_z_2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride= 2, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.conv_z_3 = nn.ConvTranspose2d(32, imgSize[0], kernel_size=4, stride= 2, padding=1)
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
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        #h = self.relu(self.bn4(self.conv4(h)))
        #print (h4.shape)
        h = h.view(-1, self.flat_shape)
        #print (h4.shape)
        h = self.relu(self.bn5(self.fc1(h)))
        return self.bn5_1(self.fc21(h)), self.bn5_2(self.fc22(h))

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
        h = self.relu(self.bn7(self.fc4(h)))
        h = h.view(-1, 64, (self.imgSize[1]/8),(self.imgSize[1]/8))
        #print (h6.shape)
        h = self.relu(self.bn8(self.conv_z_1(h)))
        #print (h7.min())#, h7.min())
        h = self.relu(self.bn9(self.conv_z_2(h)))
        #print (h8.max())
        #h9 = self.relu(self.conv_z_3(h8))
        return self.sigmoid(self.bn10(self.conv_z_3(h)))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, *(self.imgSize)))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
