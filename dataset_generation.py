
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import csv
from feature_extraction import feature_extraction
from sklearn.preprocessing import StandardScaler


# In[2]:


scaler = StandardScaler()
folder = "/Users/tingtingx/Documents/CEG4/SEM1/CG3002/ML_Code/RawDataSet3/"
file = ['wiper','number7','chicken','sidestep','turnclap','numbersix','salute','mermaid','swing','rest','logout','cowboy']


# In[3]:


#apply overlaping sliding window
def load_segments(file_name, window_size, overlap):
    df = pd.read_csv(file_name,header =None)    
    data = df.values #change to numpy array
    data_seg = []
    label_seg = []
    for i in range(int(len(data)/overlap)-1):
        data_seg.append(data[i*overlap:((i*overlap)+(window_size-1)),0:12])
        label_seg.append(int(data[i][12]))
    return data_seg, label_seg


# In[4]:


data=[]
label=[]

for i in range(0,5):
    for j in range(1,13):        
        x,y= load_segments(folder+''.join(file[i])+str(j)+".csv",32 ,16)
        data.extend(x)
        label.extend(y)
for i in range(5,11):
    for j in range(1, 7):        
        x,y= load_segments(folder+''.join(file[i])+str(j)+".csv",32 ,16)
        data.extend(x)
        label.extend(y)
for i in range(11,12):
    for j in range(1, 12):
        x,y= load_segments(folder+''.join(file[i])+str(j)+".csv",32 ,16)
        data.extend(x)
        label.extend(y)


# In[ ]:


feature_list=[]
for i in range(int(len(data))):
    feature_list.append(feature_extraction(np.asarray(data[i])))
print(len(feature_list[0]))


# In[ ]:


df_data= pd.DataFrame(feature_list)
df_label=pd.DataFrame(label)


# In[ ]:


#save as csv
df_data.to_csv("datasetSetFinal.csv", sep=',', header=None, index=None)
df_label.to_csv("labelSetFinal.csv", header=None, index=None)

