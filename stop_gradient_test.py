from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('u/vermavik/data/DARC/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('u/vermavik/data/DARC/mnist', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)



def compute_param_norm(param):
    norm = torch.sum(param* param)
    return  norm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## encode
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1 = nn.Linear(784, 50)
        self.encoder_params=[]
        self.encoder_params= list(self.fc1.parameters())
        # transition
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(50,50)
        ## reconstruct
        self.fc3 = nn.Linear(50, 784)
        #self.fc2 = nn.Linear(50, 10)

    def encode(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.fc1(x)
        
        return x
    
    def transition(self, x):
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 320)
        x= self.fc2(x)
        return x
    
    def reconstruct(self,x):
        
        x= self.fc3(x)
        return x

def loss_z(z, z_tilda):
    return ((z-z_tilda)**2).mean()


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD( model.encoder_params, lr=args.lr, momentum=args.momentum)
loss_mse = nn.MSELoss()
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx==0:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 784)
            for i in xrange(5):
                if i==0:
                    
                    optimizer.zero_grad()
                    z = model.encode(data)
                    z_tilda = model.transition(z)
                    output = model.reconstruct(z_tilda)
                    loss = loss_mse(output, data)#loss_z(z_tilda, z)+   
                    loss.backward()
                    optimizer.step()
                    print (compute_param_norm(model.fc1.weight.data))
                    
                else:
                    optimizer.zero_grad()
                    #print (model.fc1.weight.grad.data)
                    #print (model.fc2.weight.grad.data)
                    
                    data = Variable(output.data, requires_grad= False)
                    z =  Variable(z_tilda.data, requires_grad = False)
                    z_tilda = model.transition(z)
                    output = model.reconstruct(z_tilda)
                    loss = loss_mse(output, data) #loss_z(z_tilda, z)   
                    for param in  model.fc1.parameters():
                        param.requires_grad = False
                    #print (model.fc1.weight.requires_grad)
                    #model.fc1.weight.grad.data.zero_()
                    #print (model.fc1.weight.grad.data)
                    loss.backward()
                    print (model.fc1.weight.grad.data)
                    print (model.fc2.weight.grad.data)
                    
                    optimizer.step()
                    print (compute_param_norm(model.fc1.weight.data))
                    
                
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
        """
        else:
            z_tilda = model.transition(z)
            output = model.classify(z_tilda)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            print (compute_param_norm(model.conv1.weight.data))
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
        """ 
"""
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
"""

for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test()