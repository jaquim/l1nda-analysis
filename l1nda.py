#!/usr/bin/python

# Jaquim Cadogan

from __future__ import division
from progress.bar import Bar
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import warnings


np.set_printoptions(threshold=np.nan)
warnings.filterwarnings("ignore")

file_name = 'l1nda_event_version.csv'
data = pd.read_csv(file_name)

data_last_planned = data[data['is_last_planned'] == 't']
data_last_planned.to_csv('last_planned_event_version.csv', sep=',')
print(data_last_planned.head())


# read in data set from the features file.
def read_data_linear_reg(file_name):
    # read in data
    data = pd.read_csv(file_name, sep=';')  # read in csv file as panda object
    # delete first column, as that is non-contributal to the data
    # data.drop(data.columns[[0]], axis=1, inplace=True)
    # normalize data
    # data = (data - data.mean()) / (data.max() - data.min())
    data_size = len(data)

    # initialize X matrix, and Y vector
    X, Y = list(), list()

    theta_value = 1
    theta_size = len(data.columns)
    theta_vector = np.array([[theta_value] for value in range(theta_size)])

    for index, row in data.iterrows():
                # x vector in x matrix  are pixel values to classifcate labels upon
                X.append(np.array(1, (row[:-1])))
                # y factor (classification labels)
                Y.append(np.array([row[-1]]))

    X = np.matrix(X)
    Y = np.array(Y)

    return X, Y, data_size, theta_vector


def compute_correlation(X, Y):
    correlation_vector = list()
    for column in X.T:
        correlation_vector.append(pearsonr(np.ravel(column.tolist()), np.ravel(Y.tolist()))[0])
    return correlation_vector
