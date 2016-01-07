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

pd.options.display.max_colwidth = 100

file_name = 'COMPANY_37_BRANCH_141'
input_path = './datadump/' + file_name + '.csv'
output_path = './datadump/' + file_name + '_OUTPUT.csv'

data = pd.read_csv(input_path)

data_last_planned = data[data['is_last_planned'] == 't']

start = pd.to_datetime(data_last_planned['start'])
end = pd.to_datetime(data_last_planned['end'])

data_last_planned['date'] = start.dt.strftime('%Y-%m-%d')
data_last_planned['start'] = start.dt.strftime('%H:%M:%S')
data_last_planned['end'] = end.dt.strftime('%H:%M:%S')
data_last_planned['hours'] = (end-start)
data_last_planned.to_csv(output_path, sep=',')
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
