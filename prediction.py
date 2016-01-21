import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt


# Calculates the prediction based on the coeficients of the model
# returns a list with a prediction each day, the real hours each day
# and the planned hours each day, it also returns a list with each days date
def calc_pred(data_worked, data_planned, coef_list):

    # Create the lists to be filled later
    prediction_list, hours_list, planned_list, date_list = list(), list(), list(), list()
    for element_planned in data_planned.iterrows():
        for element_worked in data_worked.iterrows():
            # Check wether there is a planned and a worked schedule that day:
            if element_planned[1]['date'] == element_worked[1]['date']:
                date_list.append(np.ravel(element_planned)[0])
                # Create the predicted hours based on the coeficients returned by the linear regression model
                prediction = coef_list['festivity'] * element_planned[1]['festivity'] + coef_list['weather_grades'] * element_planned[1]['weather_grades'] + coef_list['last_10_weekdays'] * element_planned[1]['last_10_weekdays'] + coef_list['mean_weekday_lastyear'] * element_planned[1]['mean_weekday_lastyear'] + coef_list['lastweek_worked_hours'] * element_planned[1]['lastweek_worked_hours'] + coef_list['last_year_worked_hours'] * element_planned[1]['last_year_worked_hours'] + coef_list['theta_vector']
                # Append the predicted hours, worked hours and planned hours
                # to the lists to be returned
                prediction_list.append(prediction)
                hours_list.append(element_worked[1]['hours'])
                planned_list.append(element_planned[1]['hours'])
    prediction_model = str(coef_list['festivity']) + ' * X_festivity + ' + str(coef_list['weather_grades']) + ' * X_weathergrades + ' + str(coef_list['last_10_weekdays']) + ' * X_last_10_weekdays + ' + str(coef_list['mean_weekday_lastyear']) + ' * X_mean_weekday_lastyear + ' + str(coef_list['lastweek_worked_hours']) + ' * X_last_week_workedhours + ' + str(coef_list['last_year_worked_hours']) + ' * X_last_year_workedhours + '  + str(coef_list['theta_vector']) + ' * theta_vector'
    return prediction_list, hours_list, planned_list, date_list, prediction_model


# Plots the results of the predicted hours against the real worked hours
# and the planned hours and saves this in a folder
def save_results_real(prediction_list, hours_list, planned_list, date_list, output_path):
    plt.figure(0)
    # Create x and y labels
    plt.ylabel('Hours')
    plt.xlabel('Date in days').set_visible(False)
    # Create figure title
    plt.title('Worked hours versus planned and predicted hours')
    # Plot the data
    plt.plot(planned_list, label='planner')
    plt.plot(prediction_list, label='prediction')
    plt.plot(hours_list, label='real hours')
    plt.xticks(range(len(date_list)), [])
    # Create a legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03),
          ncol=3, fancybox=True, shadow=True)
    # Save the figure
    plt.savefig(output_path + '_hours.jpg')
    # Clear the plot,
    # otherwise the next plot will still have the previous plot in it
    plt.clf()


# Plots the difference of the predicted hours against the planned worked hours
# and saves this in a folder. It does so by evaluating how much one was wrong
# in terms of how much difference there was between the real worked hourss
def save_results_difference(prediction_list, hours_list, planned_list, date_list, output_path):
    diff_prediction = np.array(hours_list) - np.array(prediction_list)
    diff_planner = np.array(hours_list) - np.array(planned_list)
    plt.figure(1)
    # Create x and y labels
    plt.ylabel('Difference in hours')
    plt.xlabel('Date in days').set_visible(False)
    # Create figure title
    plt.title('Difference between planner/predicter and the worked hours')
    # Plot the data
    plt.plot(diff_prediction, label='prediction')
    # Add a line on y = 0
    plt.axhline(0)
    plt.plot(diff_planner, label='planner')
    # Empty the date, to much so would be messy
    plt.xticks(range(len(date_list)), [])
    # Create a legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03),
          ncol=3, fancybox=True, shadow=True)
    # Save the figure
    plt.savefig(output_path + '_difference.jpg')
    # Clear the plot,
    # otherwise the next plot will still have the previous plot in it
    plt.clf()


# The main function to calculate the predictions and save the plots
def predict(data_frame, data_planned, coef_list, output_path):
    prediction_list, hours_list, planned_list, date_list, coef_model = calc_pred(data_frame, data_planned, coef_list)
    save_results_real(prediction_list, hours_list, planned_list, date_list, output_path)
    save_results_difference(prediction_list, hours_list, planned_list, date_list, output_path)

    return prediction_list, hours_list, planned_list, date_list, coef_list, coef_model
