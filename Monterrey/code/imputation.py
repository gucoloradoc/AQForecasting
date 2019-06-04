#%% Data imputation
import pandas as pd
import numpy as np


dframe=pd.read_pickle('./AMMfull.pkl')
#%% All incomplete data removal

dframe2= dframe.swaplevel('MEDIDA','ESTACION', axis=1)
dframe3= dframe['NOROESTE']
dframe3=dframe3.dropna(how='any') #56.6% of the data set dropped


## Using sklearn (Iterative Imputer) (VAn Buuren, 2011)
#%% Results from R
NOROESTEimp = pd.read_csv('NOROESTEimp_norm.csv', parse_dates=[0])
NOROESTEimp=NOROESTEimp.set_index('FECHA')

#%% Supervised learning data imputation
## Complete data partition for training and testing
dframe4= dframe3.reset_index().drop('FECHA', axis=1)
p=0.7 # Training data percentage
#np.random.seed(100)
np.random.shuffle(dframe4.values) # randomization of the training dataset IMPORTANT!!!!
N_obs=dframe4.shape[0]
N_train=int(np.round(N_obs*p))
N_val=int(np.round((N_obs-N_train)/2))
N_test=int(np.round(N_obs-N_train-N_val))

#%% Test initialy with NO2
Y_train=dframe4['NO2'][0:(N_train-1)]
Y_val=dframe4['NO2'][(N_train):(N_train+N_val-1)]
Y_test=dframe4['NO2'][(N_train+N_val):(N_train+N_val+N_test-1)]

X_train=dframe4.drop('NO2', axis=1)[0:(N_train-1)]
X_val=dframe4.drop('NO2', axis=1)[(N_train):(N_train+N_val-1)]
X_test=dframe4.drop('NO2', axis=1)[(N_train):(N_train+N_val-1)]


#%% Use a supervised data for data inputation (k nearest neigbor)

from sklearn.neighbors import KNeighborsRegressor
n_neighbors=7

neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
neigh.fit(X_train.values, Y_train.values)

#%% testing the imputer for NO2
Y_pred_val=neigh.predict(X_val)

#%%
