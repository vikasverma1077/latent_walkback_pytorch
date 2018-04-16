'''
Created on Apr 11, 2018

@author: vermavik
'''

import numpy as np

def train_classifier():
    data= np.random.randint(10, size=100) ### latent reps indexes: 1 D array
    output = np.random.randint(5, size=100) ## factor indexes : 1 D array
    classifier = np.zeros((10,2)) ## stores the majority vote class of 10 latent reps
    for i in range(10):
        idx_i = np.where(data == i)
        outputs_i = output[idx_i]
        unique, counts = np.unique(outputs_i, return_counts=True)
        max_idx = np.argmax(counts, axis=0)
        max_class = unique[max_idx]
        classifier[i,0] = i
        classifier[i,1] = max_class
        
train_classifier()