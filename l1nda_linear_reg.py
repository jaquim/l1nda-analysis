#!/usr/bin/python

import numpy as np
import pandas as pd
import l1nda
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from progress.bar import Bar
from sklearn.svm import SVR

company_37_branch_141 = l1nda.fetch_data()

poly = False

if poly == True:
    theta_size = 7
else:
    theta_size = 5

# theta values
theta_value = 1
# initial theta vectors
theta_vector = np.array([[theta_value] for value in range(theta_size)])


# cost function implementation of vector based linear regression
def cost_function(X, Y, theta_vector, data_size):
    thetaTx = np.dot(X, theta_vector)
    thetaTxminusY = np.subtract(thetaTx, Y)
    thetaTxminusYsquared = np.square(thetaTxminusY)
    return (1/float(2*data_size))*np.sum(thetaTxminusYsquared)

# hypothesis function for both linear and logistic regression,
# more about the logistic regression in the discussion below (Excercise 3)
#
# The random variable names x,y,z,q and qq are due to debugging
# wich is also a explained in the discussion below (Ex 3)
def hypothesis_func(X, theta_vector, hypothesis_function):
    if hypothesis_function == 'multiple':   
        return np.dot(X, theta_vector)
    x = np.dot(X, theta_vector)
    y = x*-1
    z = np.exp(y) 
    theta_vector_1 = np.array([[1] for value in range(z.size)])
    q = np.add(theta_vector_1, z)
#    print z.shape
#    print theta_vector_1.shape
#    print q
    qq = float(1)/q
    return qq


# vector based gradient computation
def gradient_theta(X, Y, theta_vector, data_size, hypothesis_function, alpha):
    hypothesis = hypothesis_func(X, theta_vector, hypothesis_function)
    gradient = np.multiply(np.dot(X.transpose(), np.subtract(hypothesis, Y)), (alpha/data_size))
    theta_vector = np.subtract(theta_vector, gradient)
    return theta_vector

# vector based gradien descent algorithm,
# iterates to converge theta vector, with the use of gradien_theta function.
# purpose is to minimalze costs.
def gradient_descent(X, Y, theta_vector, data_size, hypothesis_function):
    alpha = 0.01
    iteration_amount = 80000
    initial_cost = cost_function(X, Y, theta_vector, data_size)
    print('Iteration amount:', iteration_amount)
    print('Initial vector:\n', theta_vector.transpose())
    bar = Bar('Iterating:', max=iteration_amount, fill='-', suffix='%(percent).1f%% - Time remaining: %(eta)ds - Time elapsed: %(elapsed)ds')
    for i in range(iteration_amount):
        theta_vector = gradient_theta(X, Y, theta_vector, data_size, hypothesis_function, alpha)
        bar.next()
    bar.finish()
    print('Optimized vector:\n', theta_vector.transpose())
    optimized_cost = cost_function(X, Y, theta_vector, data_size)
    factor = initial_cost / optimized_cost
    print('Alpha:', alpha, 'Init cost:', initial_cost, 'Opt cost:', optimized_cost, 'Optimized by a factor of:', factor)

def multiple(X,Y):
    print np.matrix(X).shape
    print np.matrix(Y).shape
    gradient_descent(np.matrix(X), np.array(Y), theta_vector, len(X), 'multiple')


# Compute the correlation for two numpy arrays
def compute_correlation(X, Y):
    correlation_vector = list()
    for column in X.T:
        correlation_vector.append(pearsonr(np.ravel(column).tolist(), np.ravel(Y.T).tolist())[0])
    return correlation_vector


def compute_layer_correlation(data_dict):
    for type_schedule, schedule in data_dict.items():
        print(type_schedule)
        for layer_name, data_frame in schedule.items():
            print('Correlation vector for %s:' % layer_name,
                  compute_correlation(data_frame['data'], data_frame['y_vector']))


def create_linear_models(data_dict):
    for type_schedule, schedule in data_dict.items():
        print(type_schedule)
        for data_frame in schedule.values():
            x = pd.DataFrame(data_frame['data'])[1558:]
            y = pd.DataFrame(data_frame['y_vector'])[1558:]
            x = np.random.permutation(x)
            y = np.random.permutation(y)
            multiple(x,y)

compute_layer_correlation(company_37_branch_141)
create_linear_models(company_37_branch_141)



# alpha = 0.3  # learning rate
# iteration amount

# component switch to add squares of the input vaues to the dataset



