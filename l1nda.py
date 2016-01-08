#!/usr/bin/python

# Jaquim Cadogan

from __future__ import division
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import warnings
import os
import shutil
from datetime import date
import datetime

np.set_printoptions(threshold=np.nan)
warnings.filterwarnings("ignore")

pd.options.display.max_colwidth = 100


def fetch_data(file_name='COMPANY_37_BRANCH_141',
               weather_file='datadump_weercijfer.csv',
               festivity_file='Feestdagen-2016.csv'):

    datadump_dir = './datadump/' + file_name

    if os.path.exists(datadump_dir):
        shutil.rmtree(datadump_dir)

    os.mkdir('./datadump/' + file_name)
    input_path = './datadump/' + file_name + '.csv'

    data = pd.read_csv(input_path)

    data_planned = data[data['is_last_planned'] == 't']
    data_worked = data[(data['is_deleted'] == 'f') | (data['is_deleted'] == 't') \
                & (data['forward_id'] == data['event_version_id'])]

    data_worked = transform_data(data_worked, 'WORKED', weather_file, festivity_file, file_name)
    data_planned = transform_data(data_planned, 'PLANNED', weather_file, festivity_file, file_name)

    return return_data_object(data_worked), return_data_object(data_planned)


def transform_data(data_frame, output_path, weather_file, festivity_file, file_name):

    start = pd.to_datetime(data_frame['start'])
    end = pd.to_datetime(data_frame['end'])

    data_frame['date'] = start.dt.strftime('%Y-%m-%d')
    data_frame['start'] = start.dt.strftime('%H:%M:%S')
    data_frame['end'] = end.dt.strftime('%H:%M:%S')
    data_frame['hours'] = (end-start)

    return fetch_layers(data_frame, weather_file, festivity_file, output_path, file_name)


def fetch_layers(data_frame, weather_file, festivity_file, output_path, file_name):

    output_dir = './datadump/' + file_name + '/' + output_path
    os.mkdir(output_dir)

    layer_dict = dict()
    grouped = data_frame.groupby('layer_name')
    for name, group in grouped:

        layer = group.groupby('date')['hours'].sum().reset_index()

        layer = add_mean_weekday_lastyear(layer)
        layer = add_weather(layer, weather_file)
        # layer = add_festivities(layer, festivity_file)

        layer_hours = list()
        for index, entry in enumerate(layer['hours']):
            layer_hours.append(entry.total_seconds()/3600)
        layer['hours'] = layer_hours

        layer.to_csv(('%s/%s_%s_%s.csv') % (output_dir, file_name, output_path, name), sep=',')
        layer_dict[name] = layer
    return layer_dict


def add_weather(data_frame, weather_data_file):

    weather_frame = pd.read_csv('./datadump/' + weather_data_file)
    weather_frame.drop('0', axis=1, inplace=True)

    weather_grades = list()
    for index, weather_date in enumerate(weather_frame['datum']):
        for data_date in data_frame['date']:
            if weather_date == data_date:
                weather_grades.append(weather_frame['cijfer'][index])

    data_frame['weather_grades'] = weather_grades

    return data_frame


def add_last_week(layer):
    last_week_hours = list()
    for elem in layer.iterrows():
        today = elem[1]['date']
        today = datetime.datetime.strptime(today, "%Y-%m-%d").date()
        today = today.strftime('%Y,%m,%d')
        today = datetime.datetime.strptime(today, "%Y,%m,%d").date()
        offset = datetime.timedelta(days=7)
        last_weekday = today - offset 
        last_weekday = last_weekday.strftime('%Y-%m-%d')        
        hours = layer[layer['date']==last_weekday]['hours']
        hours = (hours.dt.total_seconds()/3600)
        print(hours)
        print(type(hours))
        last_week_hours.append(hours)


def add_mean_weekday_lastyear(layer):
    layer_DateTimeIndex = pd.DatetimeIndex(layer['date'])
    layer['weekday'] = layer_DateTimeIndex.weekday
    layer = layer.groupby(['weekday']).apply(custom_mean)
    layer = layer.drop('weekday', axis=1)

    return layer


def custom_mean(grp):
    # Retrieve the mean worked time on day of the week last year
    totalhours = datetime.timedelta(0)
    counter = 0
    for row in grp.iterrows():
        datum = row[1]['date']
        datum = datetime.datetime.strptime(datum, "%Y-%m-%d")
        jaar = datum.year
        if jaar == date.today().year-1:
            totalhours = totalhours + row[1]['hours']
            counter += 1

    # If last year is not available:
    jaarcounter = 2
    while counter == 0:
        for row in grp.iterrows():
            datum = row[1]['date']
            datum = datetime.datetime.strptime(datum, "%Y-%m-%d")
            jaar = datum.year

            if jaar == date.today().year-jaarcounter:
                totalhours = totalhours + row[1]['hours']
                counter += 1
        jaarcounter += 1

    mean = totalhours/counter

    grp['mean_weekday_lastyear'] = mean.total_seconds()/3600

    return grp


def add_festivities(data_frame, festivity_file):
    # feestdagen per jaar opzoek! 2016 werkt niet, vanzelfsprekend ;)
    festivities_frame = pd.read_csv('./datadump/' + festivity_file)
    festivities_binary = list()

    for index, festivity_date in enumerate(festivities_frame['datum']):
        for data_date in data_frame['date']:
            if festivity_date == data_date:
                festivities_binary.append(1)
            else:
                festivities_binary.append(0)
    data_frame['festivities'] = festivities_binary

    return data_frame


# read in data set from the features file.
def return_data_object(data_dict):
    # delete first column, as that is non-contributal to the data_dict
    # data_dict.drop(data_dict.columns[[0]], axis=1, inplace=True)
    # normalize data_dict
    # data_dict = (data_dict - data_dict.mean()) / (data_dict.max() - data_dict.min())
    # data_dict_size = len(data_dict)

    for layer_name, data_frame in data_dict.items():
        # initialize X matrix, and Y vector
        X, Y = list(), list()
        for index, row in data_frame.iterrows():
                    # x vector in x matrix  are pixel values to classifcate labels upon
                    X.append(np.array(row[1:-1]))
                    # y factor (classification labels)
                    Y.append(np.array([row[-1]]))

        X = np.matrix(X)
        Y = np.array(Y)

        data_dict[layer_name]['data'], data_dict[layer_name]['y_vector'] = X, Y


    return data_dict

fetch_data()
