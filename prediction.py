import pandas
import numpy as np
import matplotlib.pyplot as plt
import l1nda

data = l1nda.fetch_data()

company_branch = l1nda.file_name
layers = l1nda.layers


coef_festivity = 11.6234
coef_weather_grades = 0
coef_last_10_weekdays = 0.2034
cef_mean_weekday_lastyear = 0
coef_last_week_worked_hours = 0.4719
coef_last_year_worked_hours = 0
coef_lastnumber = 5.094

def read_data_log_reg(file_name, sep):
    data = pandas.read_csv(file_name, sep=sep)
    return data

def calc_pred_2(data_worked, data_planned):
    prediction_list, hours_list, planned_list, date_list = list(), list(), list(), list()
    for element_planned in data_planned:
        for element_worked in data_worked:
            if np.ravel(element_planned)[0] == np.ravel(element_worked)[0]:
                date_list.append(np.ravel(element_planned)[0])
                ##########
                # np.ravel(element_worked)[1] = festivity
                # np.ravel(element_worked)[2] = weather_grades
                # np.ravel(element_worked)[3] = last_10_weekdays
                # np.ravel(element_worked)[4] = mean_weekday_lastyear
                # np.ravel(element_worked)[5] = last_week_worked_hours
                # np.ravel(element_worked)[6] = last_year_worked_hours
                ##########
                prediction = coef_festivity * np.ravel(element_worked)[1] + coef_weather_grades * np.ravel(element_worked)[2] + coef_last_10_weekdays * np.ravel(element_worked)[3] \
                 + cef_mean_weekday_lastyear * np.ravel(element_worked)[4] + coef_last_week_worked_hours * np.ravel(element_worked)[5]\
                  + coef_last_year_worked_hours * np.ravel(element_worked)[6] +  coef_lastnumber
                prediction_list.append(prediction)
                hours_list.append(np.ravel(element_worked)[6])
                planned_list.append(np.ravel(element_planned)[6])
    return prediction_list, hours_list, planned_list, date_list

def save_results_real(prediction_list, hours_list, planned_list, date_list, file_name):
    plt.figure(0)
    plt.ylabel('Hours')
    plt.xlabel('Date in days')
    plt.title('Worked hours versus planned and predicted hours')
    plt.plot(planned_list, label='planner')
    plt.plot(prediction_list, label='prediction')
    plt.plot(hours_list, label='real hours')
    plt.xticks(range(len(date_list)), date_list)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig('./../L1nda_plots/' + file_name + '_hours.png')
    plt.clf()

def save_results_difference(prediction_list, hours_list, planned_list, date_list, file_name):
    diff_prediction = np.array(hours_list) - np.array(prediction_list)
    # print np.array(hours_list)
    # print np.array(prediction_list)
    # print diff_prediction
    diff_planner = np.array(hours_list) - np.array(planned_list)
    plt.figure(1)
    plt.ylabel('Difference in hours')
    plt.xlabel('Date in days')
    plt.title('Difference between planner/predicter and the worked hours')
    plt.plot(diff_prediction, label='prediction')
    plt.axhline(0)
    plt.plot(diff_planner, label='planner')
    plt.xticks(range(len(date_list)), date_list)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig('./../L1nda_plots/' + file_name + '_difference.png')
    plt.clf()

for layer in layers:
    data_worked = read_data_log_reg('datadump/'+ company_branch + '/WORKED/'+ company_branch + '_WORKED_' + layer + '.csv', sep=';')
    data_planned = read_data_log_reg('datadump/' + company_branch + '/PLANNED/' + company_branch +'_PLANNED_' + layer + '.csv', sep=';')
    prediction_list, hours_list, planned_list, date_list= calc_pred_2(np.matrix(data_worked), np.matrix(data_planned))
    save_results_real(prediction_list, hours_list, planned_list, date_list, company_branch + '_PLANNED_' + layer)
    save_results_difference(prediction_list, hours_list, planned_list, date_list, company_branch + '_PLANNED_' + layer)