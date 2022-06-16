import numpy as np


def normalize(features):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)