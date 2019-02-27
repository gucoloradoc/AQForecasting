# Data Tidying for Air quality forecasting

# Enter the main "Data" folder

#%%
import numpy as np
import pandas as pd
import os

#%%
print(os.getcwd())
#Should be the code folder of the project
#%%
#Ordenar el indexing de acuerdo a Ciudad, Red de medición, año?, medida

datapath='../Data/Mexico/CDMX'
sheetpath= 'PRESION/2017PA.xls'
fullsheetpath=os.path.join(datapath,sheetpath)
print(fullsheetpath)
dfpressure= pd.read_excel(fullsheetpath)
#Data frame of pandas, addinf a prefix to a row/colimn name.add_prefrix('prefix_')
#Find the sentinel values for NaN and transform them in the appropiated np.nan value
