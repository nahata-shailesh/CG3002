import numpy as np
from scipy import stats

def mean(segment):
	return np.mean(segment)

def std_dev(segment):
	return np.std(segment)

def energy(segment):
    freq_components = np.abs(np.fft.rfft(segment))
    return np.sum(freq_components ** 2) / len(freq_components)

def entropy(segment):
    freq_components = np.abs(np.fft.rfft(segment))
    return stats.entropy(freq_components, base=2)

# Extract features for all the segments
def extract_features(segments, feature_funcs):
    def extract_features(segment):
        feature_lists = [feature_func(segment) for feature_func in feature_funcs]
        return np.concatenate(feature_lists)
    return np.array([extract_features(segment) for segment in segments])