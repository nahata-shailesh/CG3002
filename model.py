import numpy as np
import feature_extraction

FEATURES_LIST = [
	feature_extraction.energy,
    feature_extraction.entropy,
    feature_extraction.mean,
    feature_extraction.stdev
]

def extract_features(segments):
	return feature_extraction.extract_features(segments, FEATURES_LIST)