# Data Tidying for Air quality forecasting

# Enter the main "Data" folder

#%%
import numpy as np
import pandas as pd
from datetime import timedelta
import os

#%%
print(os.getcwd())
#Should be the code folder of the project
#%%
#Ordenar el indexing de acuerdo a Ciudad, Red de medición, año?, medida

datapath='Data/Mexico/CDMX'
sheetpath= 'PRESION/2017PA.xls'
fullsheetpath=os.path.join(datapath,sheetpath)
print(fullsheetpath)
dfpressure= pd.read_excel(fullsheetpath)

sheetpath= 'REDMET/2017TMP.xls'
fullsheetpath=os.path.join(datapath,sheetpath)
dftemp=pd.read_excel(fullsheetpath)

#Data frame of pandas, addinf a prefix to a row/colimn name.add_prefrix('prefix_')
#Find the sentinel values for NaN and transform them in the appropiated np.nan value

#%%


#Datetime indexing
timedvec=np.vectorize(lambda x: timedelta(hours=float(x)))
dftemp.iloc[:,0]=dftemp.iloc[:,0]+pd.Series(timedvec(dftemp.iloc[:,1]))
dftemp=dftemp.set_index('FECHA')
#Reshaping to have station as value
dftemp=dftemp.drop('HORA', axis=1)
dftemp.stack().reset_index().rename(columns={'level_1': 'ESTACION',0:'T'})

#Merging by Fecha, Hora, station


#%%
