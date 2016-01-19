#!/usr/bin/python

from scipy.stats import pearsonr
from progress.bar import Bar
import numpy as np
import pandas as pd
import prediction
import statsmodels.api as sm
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


# Compute some info on the perfomance of the planner and our model
def info(prediction_list, worked_list, planned_list, layer_name, coef_list, coef_model, total_frame):

    counter = 0

    total_pred = 0
    over_planned_pred = 0
    under_planned_pred = 0

    total_planned = 0
    over_planned_planner = 0
    under_planned_planner = 0

    # Calculate the amount of missplanned hours and the mean missplanned hours
    for pred, worked, planned in zip(prediction_list, worked_list, planned_list):
        hours_wrong_pred = pred-worked
        hours_wrong_plan = planned-worked

        # Count the times of overplanning/underplanning
        if hours_wrong_pred > 0:
            over_planned_pred += 1
        else:
            under_planned_pred += 1

        if hours_wrong_plan > 0:
            over_planned_planner += 1
        else:
            under_planned_planner += 1

        total_pred = abs(hours_wrong_pred) + total_pred
        total_planned = abs(hours_wrong_plan) + total_planned
        counter += 1

    mean_pred = total_pred/counter
    mean_planner = total_planned/counter

    total_pred = 0
    total_planned = 0
    counter = 0

    # Calculate the standarddeviation

    for pred, worked, planned in zip(prediction_list, worked_list, planned_list):
        hours_wrong_pred = abs(pred-worked)
        hours_wrong_plan = abs(planned-worked)
        total_pred = total_pred + abs(hours_wrong_pred - mean_pred)
        total_planned = total_planned + abs(hours_wrong_plan - mean_pred)

        counter += 1

    std_pred = total_pred/counter
    std_planned = total_planned/counter

    performance_ratio = (1 - total_pred/total_planned) + 1

    info = pd.DataFrame(index=range(1))

    # Add the info to a dataframe and write that to csv, for future use
    info['mean_missplanned_planner'] = mean_planner
    info['mean_missplanned_prediction'] = mean_pred
    info['std_planner'] = std_planned
    info['std_prediction'] = std_pred
    info['over_planned_planner'] = over_planned_planner
    info['under_planned_planner'] = under_planned_planner
    info['over_planned_pred'] = over_planned_pred
    info['under_planned_pred'] = under_planned_pred
    # info['coef_list'] = coef_list
    info['performance_ratio'] = performance_ratio
    info['most_predicting_feature'] = max(coef_list, key=lambda x: x[1])[0]
    info['model'] = coef_model

    # Add the info to a dataframe of the info of all the layers, for future use
    total_frame = total_frame.append(info)

    layer_name = layer_name + '_overview.csv'

    info.to_csv(layer_name, sep=',', index=False)

    return total_frame


def create_linear_models():
    # input directory for JSON data
    json_dir = './datadump/json/'
    # output directory for overall statistics
    results_dir = './datadump/results/'
    # pandas dataframe with total results
    total_frame = pd.DataFrame(columns=['mean_missplanned_planner',
                                        'mean_missplanned_prediction',
                                        'std_planner',
                                        'std_prediction',
                                        'over_planned_planner',
                                        'under_planned_planner',
                                        'over_planned_pred',
                                        'under_planned_pred',
                                        'percentage',
                                        'most_predicting_feature'])

    # creation  of output directory
    if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    bar = Bar(('Appliying regression and examine statistics:'),
             max=len(os.listdir(json_dir)),
             fill='-',
             suffix='%(percent).1f%% - Time remaining: %(eta)ds - Time elapsed: %(elapsed)ds')

    # iterate through input JSON directory to apply learning algorithm
    for json_file in os.listdir(json_dir):
        print(json_file)
        info_dir = os.path.join(results_dir, os.path.splitext(json_file)[0])

        if not os.path.exists(info_dir):
                os.mkdir(info_dir)

        with open(os.path.join(json_dir, json_file)) as data:
            json_data = json.load(data)

            for schedule_type, schedule in json_data.items():
                # apply learning algorithm only to WORKED dataset
                if schedule_type == 'WORKED':
                    # for an indication where the iteration process is
                    print(len(schedule.items()))
                    for layer, data_frame in schedule.items():
                        # for an indication where the iteration process is
                        print(layer)
                        # transform data_frame from pandas to json, back to pandas frame
                        json_data[schedule_type][layer] = pd.read_json(data_frame)
                        exclude = ['date', 'hours']
                        data_frame = pd.read_json(data_frame)
                        # filter only on 2015 data

                        # data_frame = data_frame[(data_frame['date'] > '2013-12-31')]

                        # check if there is
                        if data_frame.empty:
                            continue
                        # create the dataset by excluding the date and hours
                        X = data_frame.ix[:, data_frame.columns.difference(exclude)]
                        # instantiate y vector
                        y = data_frame['hours']
                        # create/compute/fit a multivariate linear regression model
                        # no iteration is used, but the statsmodels is
                        # vector based multiplication-wise implemented
                        linear_model = sm.OLS(y, X).fit()

                        # coeficients/ parametersoutputed by the linear regression model

                        coef_list = zip(linear_model.params.index.tolist(), linear_model.params.tolist())

                        data_planned = pd.read_json(json_data['PLANNED'][layer])

                        layer_name = info_dir + '/' + layer + '/'

                        if not os.path.exists(layer_name):
                            os.mkdir(layer_name)

                        # compute plots
                        prediction_list, worked_list, planned_list, _, coef_list, coef_model = prediction.predict(data_frame, data_planned, coef_list, layer_name + layer)
                        # compute overall statistics
                        total_frame = info(prediction_list, worked_list, planned_list, layer_name + layer, coef_list, coef_model, total_frame)
            print(total_frame.describe())
            bar.next()

    overall_most_predicting = total_frame['most_predicting_feature'].value_counts().index[0]
    total_frame.describe().to_csv(results_dir + '_TOTAL_OVERVIEW', sep=',', index=False)
    bar.finish()

    # print(total_frame.describe())
create_linear_models()
