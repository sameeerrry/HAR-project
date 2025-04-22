import numpy as np


def preprocess_input(input_list, for_lstm=False):
    features = np.array(input_list)
    if for_lstm:
        # Expecting 128 x 9 = 1152 values
        features = features.reshape((1, 128, 9))
    else:
        features = features.reshape(1, -1)
    return features

