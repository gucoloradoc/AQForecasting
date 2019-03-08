
#%% 
# Neural network model
import pandas as pd
import numpy as np

df = pd.read_csv('Data/cleaned.csv', parse_dates=['FECHA'])

#%%
df= df.set_index(['FECHA', 'ESTACION'])
#for now, work with just one station
#find the stations with the most lectures.

station_lectures=df.stack().count(level='ESTACION')
max_lectures=station_lectures.max()
station_lectures[station_lectures==max_lectures] 
#For the 2017, 2018 period the 7 stations that has the most observations are
#AJM, HGM, INN, MER, MPA, TLA, XAL. 

#%%
#Temporarily will work with TLA
dfin=(df.stack()
    .reset_index()[df.stack().reset_index().ESTACION=='TLA']
    .drop('ESTACION', axis=1))

dint=dfin.pivot(index='FECHA',columns='level_2', values=0).reset_index()

dint=dint.set_index('FECHA')
dint=dint[dint.index.year == 2017]

#In 2017, 13.6% of the data is missing
dint.unstack()[dint.unstack()==-99].count()/dint.unstack().count()
cleandf=dint.unstack()[dint.unstack()!=-99]
#%%
import datetime

