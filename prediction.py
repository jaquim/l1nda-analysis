import numpy as np
import matplotlib.pyplot as plt


def calc_pred(data_worked, data_planned, coef_list):

    prediction_list, hours_list, planned_list, date_list = list(), list(), list(), list()
    for element_planned in data_planned.iterrows():
        for element_worked in data_worked.iterrows():
            if element_planned[1]['date'] == element_worked[1]['date']:
                date_list.append(np.ravel(element_planned)[0])
                prediction = coef_list[0][1] * element_planned[1]['festivity'] + coef_list[5][1] * element_planned[1]['weather_grades'] + coef_list[1][1] * element_planned[1]['last_10_weekdays'] + coef_list[4][1] * element_planned[1]['mean_weekday_lastyear'] + coef_list[3][1] * element_planned[1]['lastweek_worked_hours'] + coef_list[2][1] * element_planned[1]['last_year_worked_hours']
                prediction_list.append(prediction)
                hours_list.append(element_worked[1]['hours'])
                planned_list.append(element_planned[1]['hours'])
    return prediction_list, hours_list, planned_list, date_list


def save_results_real(prediction_list, hours_list, planned_list, date_list, output_path):
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
    plt.savefig(output_path + '_hours.png')
    plt.clf()


def save_results_difference(prediction_list, hours_list, planned_list, date_list, output_path):
    diff_prediction = np.array(hours_list) - np.array(prediction_list)
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
    plt.savefig(output_path + '_difference.png')
    plt.clf()


def predict(data_frame, data_planned, coef_list, output_path):
    prediction_list, hours_list, planned_list, date_list = calc_pred(data_frame, data_planned, coef_list)
    save_results_real(prediction_list, hours_list, planned_list, date_list, output_path)
    save_results_difference(prediction_list, hours_list, planned_list, date_list, output_path)
