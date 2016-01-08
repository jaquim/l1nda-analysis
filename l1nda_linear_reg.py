import numpy as np
import pandas as pd
#import statsmodels.api as sm
import l1nda
from scipy.stats import pearsonr


#Compute the correlation for two numpy arrays
def compute_correlation(X, Y):
    correlation_vector = list()
    correlation_vector.append(pearsonr(np.ravel(X.tolist()), np.ravel(Y.tolist()))[0])
    print correlation_vector


data_worked, data_planned = l1nda.fetch_data()
for layer_name, data_frame in data_worked.items():
	print  layer_name
	compute_correlation(data_frame[0], data_frame[1])
