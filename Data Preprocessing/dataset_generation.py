
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv


# In[2]:


root_path= '/Users/tingtingx/Documents/CEG4/SEM1/CG3002/ML_Code/HMP_Dataset/'
folder = ['Brush_teeth/','Comb_hair/','Drink_glass/']
file = [['Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt',
       'Accelerometer-2011-04-11-13-29-54-brush_teeth-f1.txt',
       'Accelerometer-2011-05-30-21-55-04-brush_teeth-m2.txt'],
       ['Accelerometer-2011-05-30-08-32-58-comb_hair-f1.txt',
       'Accelerometer-2011-06-02-10-41-33-comb_hair-f1.txt',
       'Accelerometer-2011-06-02-16-56-03-comb_hair-f4.txt'],
       ['Accelerometer-2011-06-02-17-30-51-drink_glass-m1.txt',
       'Accelerometer-2012-03-23-03-54-54-drink_glass-m9.txt',
       'Accelerometer-2012-03-26-04-56-11-drink_glass-f2.txt']]


# In[3]:


#apply overlaping sliding window
def load_segments(file_name, activity, window_size, overlap):
    df = pd.read_table(file_name, sep=' ',header =None)
    data = df.values
    data_seg = []
    label_seg = []
    for i in range(int(len(data)/overlap)):
        data_seg.append(data[i*overlap:((i*overlap)+(window_size)),0:3])
        label_seg.append(activity)
    return data_seg, label_seg


# In[4]:


#feature extraction
def mean(segment):
    #mean of each axis, X, Y and Z
	return np.mean(segment[:,0]),np.mean(segment[:,1]),np.mean(segment[:,2])

def std_dev(segment):
    #std of each axis, X, Y and Z
	return np.std(segment[:,0]),np.std(segment[:,1]),np.std(segment[:,2])

def corr_coeff(segment):
    #correlation btw X and Y, btw X and Z, btw Y and Z
    return np.corrcoef(segment[:,0],segment[:,1])[0][1],np.corrcoef(segment[:,0],segment[:,2])[0][1],np.corrcoef(segment[:,1],segment[:,2])[0][1]

def feature_extraction(segment):
    feature=[]
    feature.extend(mean(segment))   
    feature.extend(std_dev(segment))
    feature.extend(corr_coeff(segment))
    return feature


# In[5]:


data=[]
label=[]
for i in range(int(len(folder))):
    for j in range(int(len(file[i]))):
        x,y= load_segments(root_path+''.join(folder[i])+''.join(file[i][j]),int(i+1), 200,100)
        data.extend(x)
        label.extend(y)


# In[6]:


len(data)


# In[7]:


len(label)


# In[8]:


feature_list=[]
for i in range(int(len(data))):
    feature_list.append(feature_extraction(np.asarray(data[i])))
len(feature_list)


# In[11]:


df_data= pd.DataFrame(feature_list)
df_label=pd.DataFrame(label)


# In[12]:


#save as csv
df_data.to_csv("dataset.csv", sep=',', header=None, index=None)
df_label.to_csv("labe.csv", header=None, index=None)

