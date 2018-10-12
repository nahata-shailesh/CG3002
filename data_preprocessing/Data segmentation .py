
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[11]:


def load_segments(file_name, window_size, overlap):
    df = pd.read_table(file_name, delim_whitespace=True,header =None)
    data = df.values
    segment = []
    for i in range(int(len(data)/overlap)):
        segment.append(data[i*overlap:((i*overlap)+(window_size)),0:])
    return segment 

