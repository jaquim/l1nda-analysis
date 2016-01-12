#!/usr/bin/python

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import l1nda
import statsmodels.api as sm


company_37_branch_141 = l1nda.fetch_data()

company_affiliate_name = l1nda.file_name
features = l1nda.features


# Compute the correlation for two numpy arrays
def compute_correlation(X, Y):
    correlation_vector = list()
    for column in X.T:
        correlation_vector.append(pearsonr(np.ravel(column).tolist(), np.ravel(Y.T).tolist())[0])
    return np.array(correlation_vector)


def compute_layer_correlation(data_dict):
    feature_amount = len(features)
    for type_schedule, schedule in data_dict.items():
        layer_amount = len(schedule.items())
        summed_correlation_vector = np.zeros(shape=(1, feature_amount))
        print(type_schedule + ' schedule:')
        for layer_name, data_frame in schedule.items():
            layer_correlation = compute_correlation(data_frame['data'],
                                               data_frame['y_vector'])

            summed_correlation_vector = np.add(layer_correlation,
                                               summed_correlation_vector)

            annotated_layer_correlation = zip(features, layer_correlation.tolist())

            print('Correlation for %s:\n %s' % (layer_name, annotated_layer_correlation))

        mean_correlation = np.ravel(np.divide(summed_correlation_vector, layer_amount).tolist())
        annotated_correlation = zip(features, mean_correlation)
        print('Mean correlation for %s:\n %s\n' % (company_affiliate_name, annotated_correlation))


def create_linear_models(data_dict):
    for type_schedule, schedule in data_dict.items():
        if type_schedule == 'WORKED':
            for layer_name, data_frame in schedule.items():
                X = data_frame[features]
                y = data_frame['hours']
                est = sm.OLS(y, X).fit()
                print(est.summary())

# compute_layer_correlation(company_37_branch_141)
create_linear_models(company_37_branch_141)
