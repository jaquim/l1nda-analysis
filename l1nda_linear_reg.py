#!/usr/bin/python
from __future__ import division
import matplotlib
matplotlib.use('Agg')
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

    total_pred_2 = 0
    total_planned_2 = 0
    counter = 0

    # Calculate the standarddeviation

    for pred, worked, planned in zip(prediction_list, worked_list, planned_list):
        hours_wrong_pred = abs(pred-worked)
        hours_wrong_plan = abs(planned-worked)
        total_pred_2 = total_pred_2 + abs(hours_wrong_pred - mean_pred)
        total_planned_2 = total_planned_2 + abs(hours_wrong_plan - mean_pred)

        counter += 1

    std_pred = total_pred_2/counter
    std_planned = total_planned_2/counter

    performance_ratio = total_planned/total_pred

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
    # info['model'] = str(coef_model)

    # wegschrijven per branch?
    # make a list of done things to get it back from where it stopped (log)

    # Add the info to a dataframe of the info of all the layers, for future use
    total_frame = total_frame.append(info)

    layer_name = layer_name + '_overview.csv'

    info.to_csv(layer_name, sep=',', index=False)

    return total_frame, info


# write to faulty layers to file
def write_faulty_layers(faulty_list):
    with open('faulty_layers.txt', 'w') as faulty_file:
        for layer in faulty_list:
            faulty_file.write("%s\n" % str(layer))


def create_linear_models(filter_2015):
    # input directory for JSON data
    json_dir = './datadump/json/'
    # output directory for overall statistics
    results_dir = './datadump/results/'
    # pandas dataframe with total results
    total_frame_columns = ['mean_missplanned_planner',
                           'mean_missplanned_prediction',
                           'std_planner',
                           'std_prediction',
                           'over_planned_planner',
                           'under_planned_planner',
                           'over_planned_pred',
                           'under_planned_pred',
                           'performance_ratio',
                           'most_predicting_feature']

    # overall statistics
    total_frame = pd.DataFrame(columns=total_frame_columns)
    # creation  of output directory
    if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
    os.mkdir(results_dir)
    # faulty layers
    faulty_layers = list()

    bar = Bar(('\t\t\t\tAppliying regression and examine statistics:'),
             max=len(os.listdir(json_dir)),
             fill='-',
             suffix='%(percent).1f%% - Time remaining: %(eta)ds - Time elapsed: %(elapsed)ds\n')
    # for indicative purposes where the iteration process is at
    amount_of_companies = len(os.listdir(json_dir))
    print('Amount of companies present in dataset: %s' % amount_of_companies)
    current_company = 0
    # iterate through input JSON directory to apply learning algorithm
    for json_file in os.listdir(json_dir):
        current_company += 1
        # try/except for gathering the faulty layers
        try:
            # split extension
            json_file_ex_ext = os.path.splitext(json_file)[0]
            # storage of the results
            info_dir = os.path.join(results_dir, json_file_ex_ext)
            print('\tCurrent company (#%s/%s): %s' % (current_company, amount_of_companies, json_file_ex_ext))
            # creation of the results directory
            if not os.path.exists(info_dir):
                    os.mkdir(info_dir)
            with open(os.path.join(json_dir, json_file)) as data:
                # converting current JSON file back into a dict
                json_data = json.load(data)
                # iterate through dict-like object again
                for schedule_type, schedule in json_data.items():
                    # apply learning algorithm only to WORKED dataset
                    if schedule_type == 'WORKED':
                        # for an indication where the iteration process is
                        amount_of_layers = len(schedule.items())
                        print('\t\tAmount of layers present in %s: %s' % (json_file_ex_ext, amount_of_layers))
                        # overall statistics per branch
                        branch_total_frame = pd.DataFrame(columns=total_frame_columns)
                        # for indicative measures
                        current_layer = 0
                        # store current layer
                        layer_string = str()
                        # iterate through the data_frame
                        for layer, data_frame in schedule.items():
                            # for an indication where the iteration process is
                            current_layer += 1
                            print('\t\t\tCurrent layer (#%s/%s): %s' % (current_layer, amount_of_layers, layer))
                            layer_string = layer
                            # transform data_frame from pandas to json, back to pandas frame
                            json_data[schedule_type][layer] = pd.read_json(data_frame)
                            exclude = ['date', 'hours']
                            data_frame = pd.read_json(data_frame)
                            # filter only on 2015 data
                            if filter_2015 is True:
                                data_frame = data_frame[(data_frame['date'] > '2014-12-31')]
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
                            total_frame, info_current_layer = \
                                info(prediction_list, worked_list, planned_list, layer_name + layer, coef_list, coef_model, total_frame)
                            # append current branch statistics to file
                            branch_total_frame = branch_total_frame.append(info_current_layer)
                            bar.next()
                # write current branch statistics to file
                branch_total_frame.describe().to_csv(info_dir + '/branch_statistics.csv', sep=',', index=False)
        except Exception as e:
            print(e)
            print('\t\t\t\t\tApparantly a faulty layer (skipping it): %s' % layer_string)
            # append faulty layer to all faulty layers
            faulty_layer = json_file + '_' + layer_string
            faulty_layers.append(faulty_layer)
            continue
    # write faulty layers
    write_faulty_layers(faulty_layers)
    # output overall most predicting feauture (by occurence):
    overall_most_predicting = total_frame['most_predicting_feature'].value_counts().index[0]
    print('The overall most predicting feature is: %s' % overall_most_predicting)
    # write overall statistics
    # total_frame.describe().to_csv(results_dir + 'l1nda_TOTAL_OVERVIEW', sep=',', index=False)
    # end progressbar
    bar.finish()

create_linear_models(filter_2015=False)
