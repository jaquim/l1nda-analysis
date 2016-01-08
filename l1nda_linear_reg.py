import numpy as np
import pandas as pd
#import statsmodels.api as sm
import l1nda
from scipy.stats import pearsonr



data_worked, data_planned = l1nda.fetch_data()


# def read_data():
# 	data = pd.read_csv('./datadump/COMPANY_37_BRANCH_141/WORKED/COMPANY_37_BRANCH_141_WORKED_layer432.csv')
# 	hours = np.array(data['hours'].tolist())
# 	weather_grade = np.array(data['weather_grade'].tolist())
# 	return hours, weather_grade


#Compute the correlation for two numpy arrays
def compute_correlation(X, Y):
    correlation_vector = list()
    correlation_vector.append(pearsonr(np.ravel(X.tolist()), np.ravel(Y.tolist()))[0])
    return correlation_vector


hours, weather_grade = read_data()
print compute_correlation(hours, weather_grade)