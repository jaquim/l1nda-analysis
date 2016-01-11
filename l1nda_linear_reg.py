#!/usr/bin/python

import numpy as np
import pandas as pd
import statsmodels.api as sm
import l1nda
from scipy.stats import pearsonr

company_37_branch_141 = l1nda.fetch_data()


# Compute the correlation for two numpy arrays
def compute_correlation(X, Y):
    correlation_vector = list()
    for column in X.T:
        correlation_vector.append(pearsonr(np.ravel(column).tolist(), np.ravel(Y.T).tolist())[0])
    return correlation_vector


def compute_layer_correlation(data_dict):
    for type_schedule, schedule in data_dict.items():
        print(type_schedule)
        for layer_name, data_frame in schedule.items():
            print('Correlation vector for %s:' % layer_name,
                  compute_correlation(data_frame['data'], data_frame['y_vector']))


def create_linear_models(data_dict):
    for type_schedule, schedule in data_dict.items():
        print(type_schedule)
        for data_frame in schedule.values():
            x = pd.DataFrame(data_frame['data'])
            y = pd.DataFrame(data_frame['y_vector'])
            print(x, y)
            est = sm.OLS(y, x).fit()
            print(est.summary())


compute_layer_correlation(company_37_branch_141)
# create_linear_models(company_37_branch_141)
