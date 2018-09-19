import pandas as pd
import numpy as np

#This function returns a list, each element in the list
#contains data in specified window size

def load_segments(file_name, window_size, overlap):
    df = pd.read_table(file_name, sep=',', header =None)
    data = df.values
    segment = []
    for i in range(int(len(data)/overlap)):
        segment.append(data[i*overlap:((i*overlap)+(window_size)),0:])
    return np.array(segment) 