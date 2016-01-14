#!/usr/bin/python

from __future__ import division
import numpy as np
import pandas as pd
import warnings
import os
import shutil
from progress.bar import Bar
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

# Dont throw warnings that are to be ignored anyway
warnings.filterwarnings("ignore")
# Force pandas to print full width tables
pd.options.display.max_colwidth = 100
# Force numpy to print full  length  arrays/matrices
np.set_printoptions(threshold=np.nan)

# Input data
weather_file = 'datadump_weercijfer.csv'
festivity_file = 'datadump_feestdagen.csv'

# Input file
file_name = './datadump/l1nda_full'

# Features to be extracted
features = list()
# Layers present
layers = list()

# Option to write layers to csv
write_to_csv = True

# For smaller test sets:
filter_2015 = False


def fetch_data():
    global file_name

    data = pd.read_csv(file_name + '.csv')

    grouped = data.groupby(['domain', 'branch'])

    bar = Bar(('Composing layers by computing features for %s schedule: '),
             max=len(grouped),
             fill='-',
             suffix='%(percent).1f%% - Time remaining: %(eta)ds - Time elapsed: %(elapsed)ds')

    for name, data in grouped:
        file_name = 'COMPANY_' + str(name[0]) + '_BRANCH_' + str(name[1])

        # Create directory for the data to be placed in
        datadump_dir = './datadump/' + file_name

        if write_to_csv is True:
            # Check if datadump_dir exists, if so, remove
            if os.path.exists(datadump_dir):
                shutil.rmtree(datadump_dir)
            os.mkdir('./datadump/' + file_name)

        # Retrieve from file planned working hours schedule
        data_planned = data[data['is_last_planned'] == 't']
        # Retrieve actual worked hours, based on either 'is_deleted'
        # Boolean is set to true, or forward_id is equal to the
        # Id of the scheme itself. This can only occur if the  'is_last_planned'
        # Boolean is set true
        data_worked = data[(data['is_deleted'] == 'f') | (data['is_deleted'] == 't') \
                    & (data['forward_id'] == data['event_version_id'])]

        data_worked = transform_data(data_worked, 'WORKED')
        data_planned = transform_data(data_planned, 'PLANNED')

        company_data = {'WORKED': data_worked,
                        'PLANNED': data_planned}

        print(file_name + '\n')
        print('\nPresent features: \n%s\nPresent layers (%s):\n%s\n'
              % (features, len(layers), layers))

        bar.next()
    bar.finish()

    print('Done!')


# Calculate the actual hours per shift, which we will use to calculate
# the amount of hours per day
def transform_data(data_frame, output_path):

    global filter_2015

    start = pd.to_datetime(data_frame['start'])
    end = pd.to_datetime(data_frame['end'])
    # Transform date, start, and end columns entries from
    # Database csv dump into datetimeobjects
    data_frame['date'] = start.dt.strftime('%Y-%m-%d')
    data_frame['start'] = start.dt.strftime('%H:%M:%S')
    data_frame['end'] = end.dt.strftime('%H:%M:%S')
    # Calculate the time delta, resulting in the actual worked hours
    data_frame['hours'] = (end-start)

    return fetch_layers(data_frame, output_path)


# Calculate the total amount of hours per day and add all the different
# features
def fetch_layers(data_frame, output_path):

    global layers

    output_dir = './datadump/' + file_name + '/' + output_path

    if write_to_csv is True:
        os.mkdir(output_dir)

    grouped = data_frame.groupby('layer_name')

    layers = [name for name, _ in grouped]

    

    layer_dict = dict()
    for name, group in grouped:

        # Calculate the hours per date
        layer = group.groupby('date')['hours'].sum().reset_index()

        # Add all the features
        layer = add_festivities(layer)
        layer = add_weather(layer)
        layer = add_mean_weekday_last_10(layer)
        layer = add_mean_weekday_lastyear(layer)
        layer = add_historical_data(layer)
        layer = add_hours(layer)

        layer = order_layer(layer)

        # Normalize all the values
        # cols_to_norm = ['festivity', 'weather_grades', 'last_10_weekdays', 'mean_weekday_lastyear', 'lastweek_worked_hours', 'last_year_worked_hours']
        # layer[cols_to_norm] = layer[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

        # Use all the data, or only the data from 2015
        if filter_2015 is True:
            layer = layer[(layer['date'] > '2014-12-31')]
        if write_to_csv is True:
            layer.to_csv(('%s/%s_%s_%s.csv') % (output_dir, file_name, output_path, name), sep=',', index=False)
        layer_dict[name] = layer

    return layer_dict


# Orders the features, so that total hours worked/planned is the last column
def order_layer(layer):
    global features
    # Arrange columns in right order
    header = layer.columns.tolist()
    header = header[0:1] + header[2:] + header[1:2]

    features = header[1:6]
    layer = layer[header]

    return layer


# Turns the timedelta object 'hours' into floats
def add_hours(layer):
    # Transform hour column into H:S format
    layer_hours = list()
    for index, entry in enumerate(layer['hours']):
        layer_hours.append(entry.total_seconds()/3600)
    layer['hours'] = layer_hours

    return layer


# Add the weather grade to the corresponding date
def add_weather(data_frame):
    # Read in weather grades per day from data file
    weather_frame = pd.read_csv('./datadump/' + weather_file)
    weather_frame.drop('0', axis=1, inplace=True)

    # Mapping function that checks if the data occuring in the file
    # occurs in the dataframe, and if so appends the grade to a list
    weather_grades = list()
    for index, weather_date in enumerate(weather_frame['datum']):
        for data_date in data_frame['date']:
            if weather_date == data_date:
                weather_grades.append(weather_frame['cijfer'][index])

    # Insert the weather grades list retrieved from mapping to the dataframe
    data_frame['weather_grades'] = weather_grades

    return data_frame


# Adds a column with the hours of the same day last week and year per entry.
def add_historical_data(layer):
    last_week_hours = list()
    last_year_hours = list()

    # For every entry (day)
    for row in layer.iterrows():
        today = row[1]['date']
        # Changes the string to a datetime object
        today = datetime.datetime.strptime(today, "%Y-%m-%d").date()
        offset = timedelta(days=7)
        offset2 = timedelta(days=14)
        # Returns the worked hours of the same day last week
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
            hours = [row[1]['last_10_weekdays']]
        last_week_hours.append(round(hours[0], 2))

        # Does the same for last year, same day of the week

        # Changes the string to a datetime object
        day = today.weekday()
        offset = timedelta(days=365)
        offset2 = timedelta(days=372)
        # Returns the worked hours of the same day of the week last year
        last_year = today-offset
        last_year = last_year - timedelta(last_year.weekday() - day)
        last_year = (today - offset).strftime('%Y-%m-%d')
        hours = layer[layer['date'] == last_year]['hours']
        # If there was no entry, check a week earlier
        if hours.tolist() == []:
            last_year = today-offset2
            last_year = last_year - timedelta(last_year.weekday() - day)
            last_year = (today - offset).strftime('%Y-%m-%d')
            hours = layer[layer['date'] == last_year]['hours']
        # Converts the datetime to ints
        hours = (hours.dt.total_seconds()/3600)
        hours = hours.tolist()
        if hours == []:
            hours = [row[1]['last_10_weekdays']]
        last_year_hours.append(round(hours[0], 2))

    layer['lastweek_worked_hours'] = last_week_hours
    layer['last_year_worked_hours'] = last_year_hours

    return layer


# Add the mean hours of the last 10 same weekdays(E.g. last 10 fridays)
def add_mean_weekday_last_10(layer):
    layer_DateTimeIndex = pd.DatetimeIndex(layer['date'])
    layer['weekday'] = layer_DateTimeIndex.weekday
    # Split by weekday
    layer = layer.groupby('weekday').apply(custom_mean_10)
    layer = layer.drop('weekday', axis=1)

    return layer


# Retrieve the mean hours on the last 10 same weekdays
def custom_mean_10(grp):

    list_mean = list()

    mean_all = grp['hours'].mean()

    for index, row in enumerate(grp.iterrows()):
        total = datetime.timedelta(0)
        if index > 10:
            for x in range(10):
                hours = grp.iloc[index-x]['hours']
                total = total + hours

            mean = total/10
        else:
            mean = mean_all

        list_mean.append(round(mean.total_seconds()/3600, 2))

    grp['last_10_weekdays'] = list_mean

    return grp


# Add the mean hours of the same weekday last year
def add_mean_weekday_lastyear(layer):
    layer_DateTimeIndex = pd.DatetimeIndex(layer['date'])
    layer['weekday'] = layer_DateTimeIndex.weekday
    # Split by weekday
    layer = layer.groupby('weekday').apply(custom_mean)
    layer = layer.drop('weekday', axis=1)

    return layer


# Retrieve the mean worked time on day of the week last year
def custom_mean(grp):

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

    grp['mean_weekday_lastyear'] = round(mean.total_seconds()/3600, 2)

    return grp


# Add to each date whether it was a holiday or not
def add_festivities(data_frame):
    festivities_frame = pd.read_csv('./datadump/' + festivity_file)

    festivity_dates = pd.to_datetime(festivities_frame['datum'])
    festivities_frame['datum'] = festivity_dates.dt.strftime('%m-%d')

    dates = data_frame['date']
    dates = pd.to_datetime(dates)
    dates = dates.dt.strftime('%m-%d')

    festivities_binary = [0 for x in range(len(dates))]
    for index, data_date in enumerate(dates):
        for festivity_date in festivities_frame['datum']:
            if data_date == festivity_date:
                festivities_binary[index] = 1

    data_frame['festivity'] = festivities_binary

    return data_frame


# Read in data set from the features file.
def return_data_object(data_dict):
    for layer_name, data_frame in data_dict.items():
        # Initialize X matrix, and Y vector
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


def missplanned(layer_name):

    path_planned = './datadump/'+layer_name+'/PLANNED/'
    path_worked = './datadump/'+layer_name+'/WORKED/'+layer_name+'_WORKED'

    data_planned = pd.DataFrame()
    data_worked = pd.DataFrame()

    total = 0
    counter = 0
    over_planned = 0
    under_planned = 0

    for index, filename in enumerate(os.listdir(path_planned)):

        data_planned = pd.read_csv(path_planned + filename, sep=',')

        layer = filename.split("PLANNED")[1]

        data_worked = pd.read_csv(path_worked + layer, sep=',')

        over_planned = 0
        under_planned = 0

        for row in data_planned.iterrows():
            date_planned = row[1]['date']
            hours_planned = row[1]['hours']
            hours_worked = data_worked[data_worked['date'] == date_planned]['hours']
            hours_worked = hours_worked.tolist()[0]
            hours_wrong = hours_planned-hours_worked

            if hours_wrong > 0:
                over_planned += 1
            else:
                under_planned += 1

            total = abs(hours_wrong) + total
            counter += 1

        print('Days over-planned for ' + layer + ' = ' + str(over_planned))
        print('Days under-planned for ' + layer + ' = ' + str(under_planned))

    mean = total/counter

    print('Days over-planned for ' + filename + ' ' + str(over_planned))
    print('Days under-planned for ' + filename + ' ' + str(under_planned))

    counter = 0
    total = 0

    for index, filename in enumerate(os.listdir(path_planned)):

        data_planned = pd.read_csv(path_planned + filename, sep=',')

        layer = filename.split("PLANNED")[1]

        data_worked = pd.read_csv(path_worked + layer, sep=',')

        for row in data_planned.iterrows():
            date_planned = row[1]['date']
            hours_planned = row[1]['hours']
            hours_worked = data_worked[data_worked['date'] == date_planned]['hours']
            hours_worked = hours_worked.tolist()[0]
            hours_wrong = abs(hours_planned-hours_worked)

            total = total + abs(hours_wrong - mean)
            counter += 1

    standarddev = total/counter

    print('The planner for ' + layer_name + ' missplanned an average of ' + str(mean) + ' hours per day.')
    print('Standard deviation: ' + str(standarddev))

fetch_data()
