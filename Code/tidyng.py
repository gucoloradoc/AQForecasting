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

#%%
#Defining transformation functions
def longformat(datapath,sheetpath, vname):
    #Transforma el archivo de excel a fecha estacion, lectura
    fullsheetpath=os.path.join(datapath,sheetpath)
    print(fullsheetpath)
    dframe=pd.read_excel(fullsheetpath).rename(columns=str.upper)
    timedvec=np.vectorize(lambda x: timedelta(hours=float(x)))
    dframe.iloc[:,0]=dframe.iloc[:,0]+pd.Series(timedvec(dframe.iloc[:,1]))
    dframe=dframe.set_index('FECHA')
    #Reshaping to have station as value
    dframe=dframe.drop('HORA', axis=1)
    return dframe.stack().reset_index().rename(columns={'level_1': 'ESTACION', 0: vname})

def dexmerge(inlist):
    #Merge the dataframes contained in the dataframe with default formatting
    df=pd.merge(inlist[0],inlist[1],how='outer', on=['FECHA', 'ESTACION'])
    return df
listex=[]
listex.append(dftemp)
sheetpath=['PRESION/2017PA.xls',
           'REDMET/2017TMP.xls',
           'RAMA/2017CO.xls',
           'RAMA/2017NO.xls',
           'RAMA/2017NO2.xls',
           'RAMA/2017NOX.xls',
           'RAMA/2017O3.xls',
           'RAMA/2017PM10.xls',
           'RAMA/2017PM25.xls',
           'RAMA/2017PMCO.xls',
           'RAMA/2017SO2.xls',
           'RAMA/2018CO.xls',
           'RAMA/2018NO.xls',
           'RAMA/2018NO2.xls',
           'RAMA/2018NOX.xls',
           'REDMET/2017RH.xls',
           'REDMET/2017TMP.xls',
           'REDMET/2017WDR.xls',
           'REDMET/2017WSP.xls',
           'REDMET/2018RH.xls',
           'REDMET/2018TMP.xls',
           'REDMET/2018WDR.xls',
           'REDMET/2018WSP.xls']

vnames=['P', 'T', 'CO', 'NO', 'NO2', 'NOX', 'O3',
        'PM10', 'PM25','PMCO', 'SO2', 'CO','NO',
        'NO2','NOX','RH','TMP','WDR',
        'WSP','RH','TMP','WDR','WSP']

#[{0:'P'}, {0:'T'}]


#%%
#Merging by Fecha, Hora, station

longdfs=[longformat(datapath,x,vnames[i]) for i,x in enumerate(sheetpath)]
reptoconcat=(pd.DataFrame(vnames).groupby(0).size())[pd.DataFrame(vnames).groupby(0).size()].index.to_list()

#%%
[i for i, x in enumerate(longdfs) if x.columns.all()=='CO']
pd.concat([longdfs[x] for x in [2,11]], join='outer')
longdfs2=[pd.concat([longdfs[x] for x in [i for i, x in enumerate(longdfs) if x.columns.all()==y]], join='outer') for y in reptoconcat]


#%%
df=longdfs2[0]
for i in range(len(longdfs2)):
    df=pd.merge(df,longdfs2[i],how='outer', on=['FECHA', 'ESTACION'])

#%%
#Writing the above code as a list 
