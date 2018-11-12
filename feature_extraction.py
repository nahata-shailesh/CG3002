
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
from scipy import stats

#feature extraction
def mean(segment):
    #mean of each axis
    mean=[]
    for i in range(12):
        mean.append(np.mean(segment[:,i]))
    return mean

def minimum(segment):
    #minimum of each axis
    minimum=[]
    for i in range(12):
        minimum.append(np.min(segment[:,i]))
    return minimum
   
def maximum(segment):
    #maximum of each axis
    maximum=[]
    for i in range(12):
        maximum.append(np.max(segment[:,i]))
    return maximum

def std_dev(segment):
    #std of each axis
    std_dev=[]
    for i in range(12):
        std_dev.append(np.std(segment[:,i]))
    return std_dev

def corr_coeff(segment):
    #correlation btw accel and gyro
    coeff = []
#     coeff.append(np.corrcoef(segment[:,6], segment[:,7])[0][1])
#     coeff.append(np.corrcoef(segment[:,6], segment[:,8])[0][1])
#     coeff.append(np.corrcoef(segment[:,7], segment[:,8])[0][1])
#     coeff.append(np.corrcoef(segment[:,9], segment[:,10])[0][1])
#     coeff.append(np.corrcoef(segment[:,9], segment[:,11])[0][1])
#     coeff.append(np.corrcoef(segment[:,10], segment[:,11])[0][1])
    
    coeff.append(np.corrcoef(segment[:,0], segment[:,6])[0][1])
    coeff.append(np.corrcoef(segment[:,2], segment[:,8])[0][1])
    coeff.append(np.corrcoef(segment[:,4], segment[:,10])[0][1])
    return coeff

def energy(segment):
    energy = []
    for i in range(0,12):
        freq_components = np.abs(np.fft.rfft(segment[:,i]))
        energy.append(np.sum(freq_components ** 2) / len(freq_components))
    return energy

def entropy(segment):
    entropy = []
    for i in range(0,12):
        freq_components = np.abs(np.fft.rfft(segment[:,i]))
        entropy.append(stats.entropy(freq_components, base=2))
    return entropy

def feature_extraction(segment):
    feature=[]
    feature.extend(corr_coeff(segment))
    feature.extend(mean(segment)) 
    feature.extend(minimum(segment))
    feature.extend(maximum(segment))
    feature.extend(std_dev(segment))
    feature.extend(energy(segment))
    feature.extend(entropy(segment))        
    return feature

