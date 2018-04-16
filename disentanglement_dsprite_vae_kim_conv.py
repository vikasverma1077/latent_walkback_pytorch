## this script trains VAE on dsprite dataset and computes the distentanglement metric proposed by Kim et al
## TO DO : Batch norm in last layer of decoder and z-layer

'''
Created on Mar 29, 2018

@author: vermavik
'''
## source:https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb 
##        https://github.com/simplespy/SCAN-from-betaVAE/blob/master/beta-VAE/Peiyao_Sheng_beta_VAE.ipynb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import tensordataset
import os

from networks_disentanglement import *

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
   
                        
    parser.add_argument('--job_id', type=str, default='')
    parser.add_argument('--add_name', type=str, default='')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


# Load dataset
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

# Helper function to show images
def show_images_grid(imgs_, num_images=25):
  ncols = int(np.ceil(num_images**0.5))
  nrows = int(np.ceil(num_images / ncols))
  _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
  axes = axes.flatten()

  for ax_i, ax in enumerate(axes):
    if ax_i < num_images:
      ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')

def show_density(imgs):
  _, ax = plt.subplots()
  ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
  ax.grid('off')
  ax.set_xticks([])
  ax.set_yticks([])

#os.system('mkdir results && mkdir save_models')

torch.manual_seed(1)
log_interval = 1000
VAE_beta = 1.0

args = parse_args()
# Sample latents randomly
latents_train = sample_latent(size=args.train_size)
latents_test = sample_latent(size=1000)

# Select images
imgs_train = imgs[latent_to_index(latents_train)]
imgs_test = imgs[latent_to_index(latents_test)]

train_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_train).type(torch.FloatTensor))
test_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_test).type(torch.FloatTensor))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True, drop_last=True)



def loss_function(recon_x, x, mu, logvar, beta=VAE_beta, imgSize=1*64*64):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #print (logvar.exp())
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print (KLD)
    # Normalise by same number of elements as in reconstruction
    #KLD /= batch_size * imgSize
    
    return BCE + beta * KLD



model = VAE(args)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)#optim.Adagrad(model.parameters(), lr=1e-2)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.unsqueeze(1)
        #print (data.shape)
        data = data.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        #print (recon_batch.min(),recon_batch.max() )
        
        loss = loss_function(recon_batch, data, mu, logvar)
        #print (loss)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, data in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar, 0.5).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n].resize(n,1,64,64),
                                  recon_batch.view(args.batch_size, 1, 64, 64)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



latents_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
def curves(fixed_dim):
    lat_size = latents_sizes[fixed_dim]
    samples = np.zeros((lat_size, latents_sizes.size))
   

    for i, size in enumerate(latents_sizes):
        if i == fixed_dim:
            samples[:, i] = np.arange(0,lat_size,1)
        else: 
            random_x = np.random.randint(size)
            samples[:, i] = np.array([random_x]*lat_size)
    samples[:,1]=2
    indices_sampled = latent_to_index(samples)
    imgs_sampled = imgs[indices_sampled]
    #show_images_grid(imgs_sampled,lat_size)

    
    imgs_sampled = Variable(torch.from_numpy(imgs_sampled).type(torch.FloatTensor), volatile=True)
    mu, logvar = model.encode(imgs_sampled.view(-1, 64*64))
    z = model.reparameterize(mu, logvar)
    
    for i in range(10):
        plt.plot(z.data.numpy()[:,i])
    plt.ylabel('values on each dimension of z') 
    plt.xlabel(latents_names[fixed_dim]) 
    plt.show()

#for i in range(1,latents_sizes.size):
#    curves(i)




def fix_and_traverse():
    results = torch.FloatTensor(50,1,64,64).zero_()
    latent_sampled = sample_latent(size=2)
    indice_sampled = latent_to_index(latent_sampled)
    img_sampled = imgs[indice_sampled]
    show_images_grid(img_sampled,2)
    img_sampled = img_sampled[0]
    img_sampled = Variable(torch.from_numpy(img_sampled).type(torch.FloatTensor), volatile=True)
    mu, logvar = model.encode(img_sampled.view(-1, 64*64))
    z = model.reparameterize(mu, logvar)
    #print(logvar)
    origin_z = z.data.clone()
    for i,(fixed_dim, k) in enumerate(zip(range(10), range(40,50))):
        z = Variable(origin_z.clone())
        for mean in range(-2,3):
            #print(mean)
            #print(z)
            z_revise = model.reparameterize(mu*mean, logvar)
            z[0,fixed_dim] = z_revise[0,fixed_dim]
            #print(z)
            sample = model.decode(z)
            results[k] = sample.data.view(-1,64,64)
            k -= 10
    
    save_image(results,'traverse.png',nrow=10)
    print(logvar.exp())
            
        
#fix_and_traverse()       


def fix_and_show(fix_dim=-2, fix_setting=0, latent_idx=6):
    ## Fix posX latent to left
    latents_sampled = sample_latent(size=64)
    latents_sampled[:, fix_dim] = fix_setting
    indices_sampled = latent_to_index(latents_sampled)
    imgs_sampled = imgs[indices_sampled]

    imgs_sampled = Variable(torch.from_numpy(imgs_sampled).type(torch.FloatTensor), volatile=True)

    mu, logvar = model.encode(imgs_sampled.view(-1, 64*64))

    a = mu.data.numpy()
    a_arg = np.argsort(a[:, latent_idx+1])
    a = a[a_arg]
    z = model.reparameterize(mu, logvar)

    samples = model.decode(z)
    samples = torch.from_numpy(samples.data.numpy()[a_arg])
    save_image(samples.view(64, 1, 64, 64),'eval_samples.png')

#fix_and_show(3,0,6)


def get_disentanglement_score(filep):
    ### disentanglement_dsprite_lwb computation#####
    
    model = torch.load('model.pt')
    
    
    def select_fixed_latent_factor_index(latents_sizes):
        fixed_latent_factor_index = np.random.randint(low=1, high= latents_sizes.shape[0])# low=1 becaues first index is color which has only one value
        return fixed_latent_factor_index
    
    
    def get_std_over_large_subset(model, subset_size=100000):
        latents_sampled = sample_latent(size=subset_size)
        #print (latents_sampled.shape)
        indices_sampled = latent_to_index(latents_sampled)
        img_sampled = imgs[indices_sampled]
    
        img_sampled = Variable(torch.from_numpy(img_sampled).type(torch.FloatTensor).cuda(), volatile=True)
        
        mu, logvar = model.encode(img_sampled.unsqueeze(1))
        
        #print (mu.shape)
        std_each_z = mu.data.std(dim=0)
        return std_each_z
    
    std_each_z = get_std_over_large_subset(model, subset_size=1000)
    #print (std_each_z)
        
    def get_x(model, fixed_latent_factor_index, batch_size, std_each_z):
        latents_sampled = sample_latent(size=batch_size)
        
        #for i in range(batch_size):
        fixed_latent_factor_class = np.random.randint(low=0, high= latents_sizes[fixed_latent_factor_index])
        latents_sampled[:, fixed_latent_factor_index] = fixed_latent_factor_class
            
        
        indices_sampled = latent_to_index(latents_sampled)
        img_sampled = imgs[indices_sampled]
        
        
        img_sampled_a = Variable(torch.from_numpy(img_sampled).type(torch.FloatTensor).cuda(), volatile=True)
        
        mu, logvar = model.encode(img_sampled_a.unsqueeze(1))
        
        mu = mu.data
        
        #print (mu.std(dim=0))
        #print (mu)
        #print (std_each_z)
        normalized_z = mu/std_each_z
        
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
    
    num_samples = 10000
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        index = select_fixed_latent_factor_index(latents_sizes) ## select the index of the latent factor that has to be kept fixed for the batch of size L in paper
        y[i]= index
        x[i] = get_x(model, index, 100, std_each_z)##  difference defined at point 3 page 7 https://openreview.net/pdf?id=Sy2fzU9gl
        
    train_x = x
    train_y = y
    
    print (train_x.shape)
    print (train_y.shape)
    
    """
    ### create test data for classifier###
    ######################################
    
    num_samples = 10000
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)## target labels for the disentanglement_dsprite_lwb classifier
    
    for i in range(num_samples):
        index = select_fixed_latent_factor_index(latents_sizes) ## select the index of the latent factor that has to be kept fixed for the batch of size L in paper
        y[i]= index
        x[i] = get_x(model, index, 100, std_each_z)##  difference defined at point 3 page 7 https://openreview.net/pdf?id=Sy2fzU9gl
    #print (y)
    #print (x)  
    test_x = x
    test_y = y
    
    print (test_x.shape)
    print (test_y.shape)
    """
           
    def train_classifier():
        x = train_x  ### latent reps indexes: 1 D array
        y = train_y ## factor indexes : 1 D array
        
        classifier = np.zeros((args.nl,2)) ## stores the majority vote class of 10 latent reps
        for i in range(args.nl):
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
        
        return count
    
    classifier = train_classifier()
    print (classifier)
    filep.write(str(epoch)+'\n')
    filep.write(str(classifier))
    filep.write('\n')
    count = test_classifier()
    filep.write(str(count))
    filep.write('\n')



def experiment_name_vae(train_size=10000,
                    epochs=10,
                    z_size = 256,
                    job_id=None,
                    add_name=''):
    exp_name = str('vae')
    exp_name += '_train_size_'+str(train_size)
    exp_name += '_epochs_'+str(epochs)
    exp_name += '_z_size_'+str(z_size)
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)
    
    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name


## for training and saving VAE

tmp='/Tmp/vermavik/'
home='/u/vermavik/'

dataset = 'dsprite'
data_source_dir = home+'data/'+dataset+'/'

### set up the experiment directories########
exp_name = experiment_name_vae(train_size= args.train_size,
                epochs = args.epochs,
                z_size = args.nl,
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

  
for epoch in range(args.epochs):
    
    train(epoch)
    test(epoch)
    torch.save(model, model_dir+ '/model.pt')
    
    sample = Variable(torch.randn(64, args.nl)).cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 64, 64),
               result_dir+'/sample_' + str(epoch) + '.png')
    
    torch.save(model.state_dict(), '{0}/model_epoch_{1}.pth'.format(model_dir+'/', epoch))
    
    get_disentanglement_score(filep)


