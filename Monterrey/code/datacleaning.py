#%% Data Tidying for Air quality forecasting
# Monterrey DB.
# Enter the main "Data" folder

#%%
import numpy as np
import pandas as pd
from datetime import timedelta
import os

#%%
def dfretrival(datapath, sheetpath):
    fullsheetpath=os.path.join(datapath,sheetpath)
    dframe= pd.read_excel(fullsheetpath, index_col=[0,1], header=[0,1])
    dframe.index.names=['FECHA', 'HORA']
    dframe=dframe.reset_index()
    if type(dframe.HORA[0]).__name__=='time':
        timedvec=np.vectorize(lambda x: timedelta(hours=float(x.hour)))
    else:
        timedvec=np.vectorize(lambda x: timedelta(hours=float(x)))
    dframe.iloc[:,0]=dframe.iloc[:,0]+pd.Series(timedvec(dframe.iloc[:,1]))
    dframe=dframe.set_index('FECHA')
    #Reshaping to have station as value
    dframe=dframe.drop('HORA', axis=1)
    dframe.columns.names=['ESTACION', 'MEDIDA']
    return dframe

#%%
datapath='Monterrey/data/raw/'

dframe=pd.concat(
    [dfretrival(datapath,'Todo 2012.xlsx'),
    dfretrival(datapath,'Todo 2013.xlsx'),
    dfretrival(datapath,'Todo 2014.xlsx'),
    dfretrival(datapath,'Todo 2015.xlsx'),
    dfretrival(datapath,'Todo 2016.xlsx'),
    dfretrival(datapath,'Todo 2017.xlsx'),
    ])

#Random value strings in 2012 and 2013 (34 observations)
strange_strings=(dframe.applymap(type)==type('s'))
strange_strings.sum().sum()
dframe.loc[strange_strings.sum(axis=1)>0,strange_strings.sum()>0]=dframe.loc[strange_strings.sum(axis=1)>0,strange_strings.sum()>0].applymap(lambda x: np.nan if type(x)==type('s') else x)

#Some functions requiere all the values to be float
strange_ints=(dframe.applymap(type)==type(1))
dframe=dframe.applymap(np.vectorize(float))
dframe.loc[strange_ints.sum(axis=1)>0,strange_ints.sum()>0]=dframe.loc[strange_ints.sum(axis=1)>0,strange_ints.sum()>0].applymap(lambda x: float(x) if type(x)==type(1) else x)

#%% Saving the dataframe in a compressed binary file
dframe.to_pickle('./AMMfull.pkl')
#%% Visualization, and cleaning
dframe.swaplevel('MEDIDA','ESTACION', axis=1)['NOX'].plot()
dframe.columns.levels[1]
dframe.swaplevel('MEDIDA','ESTACION', axis=1)['TOUT'].iloc[:,3].hist(bins=50)

dframe.swaplevel('MEDIDA','ESTACION', axis=1)['NOX'].iloc[:,3].hist(bins=50)

dframe.swaplevel('MEDIDA','ESTACION', axis=1)['O3'].boxplot()

dframe.swaplevel('MEDIDA','ESTACION', axis=1)['NOX'].boxplot()

dframe.swaplevel('MEDIDA','ESTACION', axis=1)['TOUT'].boxplot()

dframe.swaplevel('MEDIDA','ESTACION', axis=1)['PM10'].hist(bins=50)

dframe.swaplevel('MEDIDA','ESTACION', axis=1)['2012'].count().unstack('MEDIDA')


#%%
#Subplot 4x4 de cada año. para las 15 variables medidas, todas las estaciones.

dframe.swaplevel('MEDIDA','ESTACION', axis=1)['PM10']['NOROESTE'].map(lambda x: np.log(x)).hist(bins=50)
