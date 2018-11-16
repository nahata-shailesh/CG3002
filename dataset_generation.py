
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import csv


# In[2]:


folder = "/Users/tingtingx/Documents/CEG4/SEM1/CG3002/ML_Code/train_raw_data/"
file = ['wiper','number7','chicken','sidestep','turnclap']


# In[3]:


#apply overlaping sliding window
def load_segments(file_name, window_size, overlap):
    df = pd.read_csv(file_name,header =None)
    data = df.values #change to numpy array
    data_seg = []
    label_seg = []
    for i in range(int(len(data)/overlap)):
        data_seg.append(data[i*overlap:((i*overlap)+(window_size)),0:12])
        label_seg.append(int(data[i][12]))
    return data_seg, label_seg


# In[4]:


#feature extraction
def mean(segment):
    #mean of each axis
    mean=[]
    for i in range(12):
        mean.append(np.mean(segment[:,i]))
    return mean

def std_dev(segment):
    #std of each axis
    std_dev=[]
    for i in range(12):
        std_dev.append(np.std(segment[:,i]))
    return std_dev

def corr_coeff(segment):
    #correlation btw accel and gyro
    coeff = []
    for i in range(0,3):
        coeff.append(np.corrcoef(segment[:,i], segment[:,i+3])[0][1])
    for i in range(6,9):
        coeff.append(np.corrcoef(segment[:,i], segment[:,i+3])[0][1])
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
    feature.extend(mean(segment))   
    feature.extend(std_dev(segment))
    feature.extend(corr_coeff(segment))
    feature.extend(energy(segment))
    feature.extend(entropy(segment))        
    return feature


# In[5]:


data=[]
label=[]

for i in range(len(file)):
    for j in range(1,7):        
        x,y= load_segments(folder+''.join(file[i])+str(j)+".csv",40 ,20)
        data.extend(x)
        label.extend(y)


# In[6]:





# In[7]:





# In[11]:


feature_list=[]
for i in range(int(len(data))):
    feature_list.append(feature_extraction(np.asarray(data[i])))


# In[9]:


df_data= pd.DataFrame(feature_list)
df_label=pd.DataFrame(label)


# In[10]:


#save as csv
df_data.to_csv("dataset.csv", sep=',', header=None, index=None)
df_label.to_csv("label.csv", header=None, index=None)

