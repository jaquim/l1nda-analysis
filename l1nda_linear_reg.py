#!/usr/bin/python

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import l1nda
import statsmodels.api as sm
import matplotlib.pyplot as plt
import prediction
import os
import json

# numpy, pandas = l1nda.fetch_data()

# company_affiliate_name = l1nda.file_name
# features = l1nda.features


# Compute the correlation for two numpy arrays
def compute_correlation(X, Y):
    correlation_vector = list()
    for column in X.T:
        correlation_vector.append(pearsonr(np.ravel(column).tolist(), np.ravel(Y.T).tolist())[0])
    return np.array(correlation_vector)


def compute_layer_correlation(data_dict, features, company_affiliate_name):
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


def create_linear_models():
    json_dir = './datadump/json/'

    for json_file in os.listdir(json_dir):
        with open(os.path.join(json_dir, json_file)) as data:
            json_data = json.load(data)
            print json_file
            for schedule_type, schedule in json_data.items():
                if schedule_type == 'WORKED':
                    for layer, data_frame in schedule.items():
                        # json_data[schedule_type][layer] = pd.read_json(data_frame)
                        exclude = ['date', 'hours']
                        data_frame = pd.read_json(data_frame)
                        data_frame = data_frame[(data_frame['date'] > '2014-12-31')]
                        X = data_frame.ix[:, data_frame.columns.difference(exclude)]
                        y = data_frame['hours']

                        print layer
                        est = sm.OLS(y, X).fit()
                        coef_list = (zip(est.params.index.tolist(), est.params.tolist()))
                        data_planned = pd.read_json(json_data['PLANNED'][layer])
                        company_branch = os.path.splitext(json_file)[0]
                        prediction_list, hours_list, planned_list, date_list = prediction.calc_pred(data_frame, data_planned, coef_list)
                        prediction.save_results_real(prediction_list, hours_list, planned_list, date_list, company_branch + '_PLANNED_' + layer)
                        prediction.save_results_difference(prediction_list, hours_list, planned_list, date_list, company_branch + '_PLANNED_' + layer)
                        break
                    break
                break
            break

create_linear_models()
