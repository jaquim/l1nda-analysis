#!/usr/bin/python

import numpy as np
import pandas as pd
#import statsmodels.api as sm
import l1nda
from scipy.stats import pearsonr

data_worked, data_planned = l1nda.fetch_data()


#Compute the correlation for two numpy arrays
def compute_correlation(X, Y):
    correlation_vector = list()
    correlation_vector.append(pearsonr(np.ravel(X.tolist()), np.ravel(Y.tolist()))[0])
    return correlation_vector


def compute_layer_correlation(data_dict):
    for layer_name, data_frame in data_dict.items():
        layer_correlation = compute_correlation(data_frame['data'], data_frame['y_vector'])
        print(layer_correlation)

compute_layer_correlation(data_worked)
