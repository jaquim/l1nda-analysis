#!/usr/bin/python

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import l1nda
import prediction
import statsmodels.api as sm
import prediction
import os
import shutil
import json


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


def info(data_planned, data_worked, info_dir, coef_list, total_frame):

    prediction_list, worked_list, planned_list, date_list = prediction.calc_pred(data_worked, data_planned, coef_list)

    # Prediction

    total_pred = 0
    counter = 0
    over_planned_pred = 0
    under_planned_pred = 0

    for pred, worked in zip(prediction_list, worked_list):
        hours_wrong = pred-worked

        if hours_wrong > 0:
            over_planned_pred += 1
        else:
            under_planned_pred += 1

        total_pred = abs(hours_wrong) + total_pred
        counter += 1

    mean_pred = total_pred/counter

    total = 0
    counter = 0

    for pred, worked in zip(prediction_list, worked_list):
        hours_wrong = abs(pred-worked)
        total = total + abs(hours_wrong - mean_pred)
        counter += 1

    std_pred = total/counter

    # Planned

    total_planned = 0
    counter = 0
    over_planned_planner = 0
    under_planned_planner = 0
    for planned, worked in zip(planned_list, worked_list):
        hours_wrong = planned-worked

        if hours_wrong > 0:
            over_planned_planner += 1
        else:
            under_planned_planner += 1

        total_planned = abs(hours_wrong) + total_planned
        counter += 1

    mean_planner = total_planned/counter

    total = 0
    counter = 0

    for planned, worked in zip(planned_list, worked_list):
        hours_wrong = abs(planned-worked)
        total = total + abs(hours_wrong - mean_planner)
        counter += 1

    std_planned = total/counter

    percentage = total_pred/total_planned * 100

    info = pd.DataFrame(index=range(1))

    info['mean_missplanned_planner'] = mean_planner
    info['mean_missplanned_prediction'] = mean_pred
    info['std_planner'] = std_planned
    info['std_prediction'] = std_pred
    info['over_planned_planner'] = over_planned_planner
    info['under_planned_planner'] = under_planned_planner
    info['over_planned_pred'] = over_planned_pred
    info['under_planned_pred'] = under_planned_pred
    # info['coef_list'] = coef_list
    info['percentage'] = percentage
    info['most_predicting_feature'] = max(coef_list, key=lambda x: x[1])[0]

    total_frame = total_frame.append(info)

    layer_name = info_dir + '_overview.csv'

    info.to_csv(layer_name, sep=',', index=False)


def create_linear_models():
    json_dir = './datadump/json/'
    results_dir = './datadump/results/'
    total_frame = pd.DataFrame()
    if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    for json_file in os.listdir(json_dir):
        print(json_file)
        info_dir = os.path.join(results_dir, os.path.splitext(json_file)[0])

        if not os.path.exists(info_dir):
                os.mkdir(info_dir)

        with open(os.path.join(json_dir, json_file)) as data:
            json_data = json.load(data)

            for schedule_type, schedule in json_data.items():
                if schedule_type == 'WORKED':
                    print(len(schedule.items()))
                    for layer, data_frame in schedule.items():
                        print(layer)
                        json_data[schedule_type][layer] = pd.read_json(data_frame)
                        exclude = ['date', 'hours']
                        data_frame = pd.read_json(data_frame)

                        data_frame = data_frame[(data_frame['date'] > '2014-12-31')]

                        X = data_frame.ix[:, data_frame.columns.difference(exclude)]
                        y = data_frame['hours']
                        est = sm.OLS(y, X).fit()

                        coef_list = zip(est.params.index.tolist(), est.params.tolist())

                        data_planned = pd.read_json(json_data['PLANNED'][layer])

                        layer_name = info_dir + '/' + layer + '/'

                        if not os.path.exists(layer_name):
                            os.mkdir(layer_name)

                        prediction.predict(data_frame, data_planned, coef_list, layer_name + layer)
                        info(data_planned, data_frame, layer_name + layer, coef_list, total_frame)
                        print(total_frame.describe())

    print(total_frame.describe())
create_linear_models()
