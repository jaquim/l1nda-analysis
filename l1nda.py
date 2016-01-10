#!/usr/bin/python

from __future__ import division
import numpy as np
import pandas as pd
import warnings
import os
import shutil
from datetime import date, timedelta
import datetime

_author_ = "Simon Hoekman, Wout Kooijman, Max Mijnheer, Jaquim Cadogan"

"""
    Use:
        ```python
           import l1nda
           company_data = l1nda.fetch_data()
        ```
    Ouput to dict (.csv) in the following format:

    {WORKED: {
            layer[id]:{
                'data': numpy.matrix
                'y_vector': nump.ndarray
                },
            ...
            ...
        },
    PLANNED: {
            layer[id]:{
                'data': numpy.matrix
                'y_vector': nump.ndarray
                },
            ...
            ...
        },
    }
"""

# dont throw warnings that are to be ignored anyway
warnings.filterwarnings("ignore")
# force pandas to print full width tables
pd.options.display.max_colwidth = 100
# force numpy to print full  length  arrays/matrices
np.set_printoptions(threshold=np.nan)

# input data
weather_file = 'datadump_weercijfer.csv'
festivity_file = 'datadump_feestdagen.csv'

# input file
file_name = 'COMPANY_37_BRANCH_141'


def fetch_data():

    # create directory for the data to be placed in
    datadump_dir = './datadump/' + file_name

    # check if datadump_dir exists, if so, remove
    if os.path.exists(datadump_dir):
        shutil.rmtree(datadump_dir)

    # read in data file and create datadump_dir named according file_name
    input_path = './datadump/' + file_name + '.csv'
    os.mkdir('./datadump/' + file_name)

    # instantiate pandas dataframe
    data = pd.read_csv(input_path)

    # retrieve from file planned working hours schedule
    data_planned = data[data['is_last_planned'] == 't']
    # retrieve actual worked hours, based on either 'is_deleted'
    # boolean is set to true, or  forward_id is equal to the 
    # id of the scheme itself. This can only occur if the  'is_last_planned'
    # boolean is set true
    data_worked = data[(data['is_deleted'] == 'f') | (data['is_deleted'] == 't') \
                & (data['forward_id'] == data['event_version_id'])]

    data_worked = transform_data(data_worked, 'WORKED')
    data_planned = transform_data(data_planned, 'PLANNED')

    company_data = {'WORKED': return_data_object(data_worked),
                    'PLANNED': return_data_object(data_planned)}

    return company_data


def transform_data(data_frame, output_path):

    start = pd.to_datetime(data_frame['start'])
    end = pd.to_datetime(data_frame['end'])
    # transform date, start, and end columns entries from
    # database csv dump into datetimeobjects
    data_frame['date'] = start.dt.strftime('%Y-%m-%d')
    data_frame['start'] = start.dt.strftime('%H:%M:%S')
    data_frame['end'] = end.dt.strftime('%H:%M:%S')
    # calculate the time delta, resulting in the actual worked hours
    data_frame['hours'] = (end-start)

    return fetch_layers(data_frame, output_path)


def fetch_layers(data_frame, output_path):

    output_dir = './datadump/' + file_name + '/' + output_path
    os.mkdir(output_dir)

    layer_dict = dict()
    grouped = data_frame.groupby('layer_name')
    for name, group in grouped:

        layer = group.groupby('date')['hours'].sum().reset_index()

        # layer = add_festivities(layer)
        layer = add_weather(layer)
        layer = add_mean_weekday_lastyear(layer)
        layer = add_last_week(layer)
        layer = add_hours(layer)

        layer.to_csv(('%s/%s_%s_%s.csv') % (output_dir, file_name, output_path, name), sep=';')
        layer_dict[name] = layer
    return layer_dict


def add_hours(layer):
    # transform hour column into H:S format
    layer_hours = list()
    for index, entry in enumerate(layer['hours']):
        layer_hours.append(entry.total_seconds()/3600)
    layer['hours'] = layer_hours

    return layer


def add_weather(data_frame):
    # read in weather grades per day from data file
    weather_frame = pd.read_csv('./datadump/' + weather_file)
    weather_frame.drop('0', axis=1, inplace=True)

    # mapping function that checks if the data occuring in the file
    # occurs in the dataframe, and if so appends the grade to a list
    weather_grades = list()
    for index, weather_date in enumerate(weather_frame['datum']):
        for data_date in data_frame['date']:
            if weather_date == data_date:
                weather_grades.append(weather_frame['cijfer'][index])

    # insert the weather grades list retrieved from mapping to the dataframe
    data_frame['weather_grades'] = weather_grades

    return data_frame


# Adds a column with the worked hours of the same day last week per entry.
def add_last_week(layer):
    last_week_hours = list()
    # for every entry (day)
    for today in layer['date']:
        # changes the string to a datetime object
        today = datetime.datetime.strptime(today, "%Y-%m-%d").date()
        offset = timedelta(days=7)
        offset2 = timedelta(days=14)
        # returns the worked hours of the same day last week
        last_weekday = (today - offset).strftime('%Y-%m-%d')
        hours = layer[layer['date'] == last_weekday]['hours']
        # If there was no entry last week, check 2 weeks ago
        if hours.tolist() == []:
            last_weekday = today - offset2
            last_weekday = last_weekday.strftime('%Y-%m-%d')
            hours = layer[layer['date'] == last_weekday]['hours']
        # Converts the datetime to ints
        hours = (hours.dt.total_seconds()/3600)
        hours = hours.tolist()
        if hours == []:
            hours.append(0)
        last_week_hours.append(round(hours[0], 2))

    layer['lastweek_worked_hours'] = last_week_hours

    return layer


def add_mean_weekday_lastyear(layer):
    layer_DateTimeIndex = pd.DatetimeIndex(layer['date'])
    layer['weekday'] = layer_DateTimeIndex.weekday
    layer = layer.groupby('weekday').apply(custom_mean)
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


def add_festivities(data_frame):
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
        print('A')
        # initialize X matrix, and Y vector
        X, Y = list(), list()
        for index, row in data_frame.iterrows():
                    # x vector in x matrix  are pixel values to classifcate labels upon
                    X.append(np.array(row[1:-1]))
                    # y factor (classification labels)
                    Y.append(np.array([row[-1]]))

        X = np.matrix(X)
        Y = np.array(Y)

        data_dict[layer_name] = {'data': X, 'y_vector': Y}

    return data_dict
