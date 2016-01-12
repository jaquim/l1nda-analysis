import pandas
import numpy as np
import matplotlib.pyplot as plt

layer = 'layer928'
company_branch = 'COMPANY_45_BRANCH_325'


def read_data_log_reg(file_name, sep):
    data = pandas.read_csv(file_name, sep=sep)
    return data


def calc_pred(data_worked, data_planned, data_date):
    prediction_list, hours_list = list(), list()
    for row in data_worked.iterrows():
        prediction = 0.8755 * row[1]['mean_weekday_lastyear'] + 0.1273 * row[1]['last_week_worked_hours'] - 0.1166
        prediction_list.append(prediction)
        hours_list.append(row[1]['hours'])
    planned_list, date_list = list(), list()
    for row in data_planned.iterrows():
        planned_list.append(row[1]['hours'])
    for row in data_date.iterrows():
        date_list.append(row[1]['date'])
    return prediction_list, hours_list, planned_list, date_list

def plot_results_real(prediction_list, hours_list, planned_list, date_list, file_name):
    plt.figure(0)
    plt.ylabel('Hours')
    plt.xlabel('Date in days')
    plt.title('Worked hours versus planned and predicted hours')
    plt.plot(planned_list, label='planner')
    plt.plot(prediction_list, label='prediction')
    plt.plot(hours_list, label='real hours')
    plt.xticks(range(len(date_list)), date_list)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('./../L1nda_plots/' + file_name + '_hours.png')

def plot_results_difference(prediction_list, hours_list, planned_list, date_list, file_name):
    diff_prediction = np.array(hours_list) - np.array(prediction_list)
    # print np.array(hours_list)
    # print np.array(prediction_list)
    # print diff_prediction
    diff_planner = np.array(hours_list) - np.array(planned_list)
    plt.figure(1)
    plt.ylabel('Hours')
    plt.xlabel('Date in days')
    plt.title('Difference between planner/predicter and the worked hours')
    plt.plot(diff_prediction, label='prediction')
    plt.axhline(0)
    plt.plot(diff_planner, label='planner')
    plt.xticks(range(len(date_list)), date_list)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('./../L1nda_plots/' + file_name + '_difference.png')


data_worked = read_data_log_reg('datadump/'+ company_branch + '/WORKED/'+ company_branch + '_WORKED_' + layer + 'test.csv', sep=',')
data_planned = read_data_log_reg('datadump/' + company_branch + '/PLANNED/' + company_branch +'_PLANNED_' + layer + 'test.csv', sep=',')
data_date = read_data_log_reg('datadump/' + company_branch + '/PLANNED/' + company_branch + '_PLANNED_' + layer + '.csv', sep =',')
prediction_list, hours_list, planned_list, date_list= calc_pred(data_worked, data_planned, data_date)
plot_results_real(prediction_list, hours_list, planned_list, date_list, company_branch + '_PLANNED_' + layer)
plot_results_difference(prediction_list, hours_list, planned_list, date_list, company_branch + '_PLANNED_' + layer)