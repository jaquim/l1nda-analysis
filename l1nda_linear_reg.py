import numpy as numpy
import pandas as pd
import statsmodels.api as sm
import l1nda



def compute_correlation(X, Y):
    correlation_vector = list()
    for column in X.T:
        correlation_vector.append(pearsonr(np.ravel(column.tolist()), np.ravel(Y.tolist()))[0])
    return correlation_vector


print compute_correlation(weather_grade, worked_hours)
