from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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




# Change figure aesthetics
#%matplotlib inline
#ssns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 1.5})

# Load dataset
dataset_zip = np.load('/u/vermavik/github/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

print('Keys in the dataset:', dataset_zip.keys())
imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']

metadata = dataset_zip['metadata'][()]

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata['latents_sizes']
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





os.system('mkdir results && mkdir save_models')

torch.manual_seed(1)
batch_size = 128
log_interval = 10
epochs = 100
VAE_beta = 4.0

# Sample latents randomly
latents_train = sample_latent(size=5000)
latents_test = sample_latent(size=1000)

# Select images
imgs_train = imgs[latent_to_index(latents_train)]
imgs_test = imgs[latent_to_index(latents_test)]

train_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_train).type(torch.FloatTensor))
test_dataset = tensordataset.TensorDataset(torch.from_numpy(imgs_test).type(torch.FloatTensor))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def select_fixed_latent_factor_index(latents_sizes):
    fixed_latent_factor_index = np.random.randint(low=0, high= latents_sizes.shape[0])
    return fixed_latent_factor_index
    
def sample_latent(size=1):
  samples = np.zeros((size, latents_sizes.size))
  for lat_i, lat_size in enumerate(latents_sizes):
    samples[:, lat_i] = np.random.randint(lat_size, size=size)

  return samples


def get_batch_samples(fixed_latent_factor_index,fixed_latent_factor_class, batch_size):
    ##step 2 on page 7 : https://openreview.net/pdf?id=Sy2fzU9gl
    #for i in range(batch_size):
    
    latents_sampled_a = sample_latent(size=batch_size)
    latents_sampled_b = sample_latent(size=batch_size)
    
    for i in range(batch_size):
        fixed_latent_factor_class = np.random.randint(low=0, high= latents_sizes[fixed_latent_factor_index]) 
        latents_sampled_a[i, fixed_latent_factor_index] = fixed_latent_factor_class
        latents_sampled_b[i, fixed_latent_factor_index] = fixed_latent_factor_class
    
    indices_sampled_a = latent_to_index(latents_sampled_a)
    img_sampled_a = imgs[indices_sampled_a]

    indices_sampled_b = latent_to_index(latents_sampled_b)
    img_sampled_b = imgs[indices_sampled_b]
    
    img_sampled_a = Variable(torch.from_numpy(img_sampled_a).type(torch.FloatTensor), volatile=True)
    img_sampled_b = Variable(torch.from_numpy(img_sampled_b).type(torch.FloatTensor), volatile=True)

    mu_a, logvar_a = model.encode(img_sampled_a.view(-1, 64*64))
    mu_b, logvar_b = model.encode(img_sampled_b.view(-1, 64*64))
    
    
    ### convert to numpy and back to tensor
    diff = nn.L1Loss()
    
    mean
    ##
    return
        
def train_classifier(num_batches):
    
    x = np.zeros(num_batches, num_z)
    y = np.zeros(num_batches)
    
    
    for i in range(num_batches):
        y[i] = select_fixed_latent_factor_index(latents_sizes)
        x[i,:] = get_batch_samples(fixed_latent_factor_index,fixed_latent_factor_class, batch_size)##  difference defined at point 3 page 7 https://openreview.net/pdf?id=Sy2fzU9gl

    
    
    
"""    