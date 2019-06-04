#%% Myth busters: Qualitative 

#%%Loading data and libraries
import matplotlib.pyplot as plt
import pandas as pd

dframe=pd.read_pickle('./AMMfull.pkl')

stations=dframe.columns.levels[0].to_list()
measures=dframe.columns.levels[1].to_list()

#Counting the number of observation per station 
dframe.loc[:,stations[1]].loc[:,'PM2.5'].count()
#%% Plotting PM2.5 in all the stations, hourly concentrations

fig, axes= plt.subplots(5,3, figsize=[20, 20])

k=0
for i in range(0,5):
    for j in range(0,3):
        if k<13:
            y,x,_ = axes[i,j].hist((dframe.loc[:,stations[k]].loc[:,'PM2.5'].dropna()), 
                bins=50, label=stations[k])
            x_max=dframe.loc[:,stations[k]].loc[:,'PM2.5'].dropna().max()
            axes[i,j].legend(loc='upper right')
            print(x.max())
            if dframe.loc[:,stations[k]].loc[:,'PM2.5'].dropna().count()>0:
                axes[i,j].arrow(x_max,y.max()/10,0,-y.max()/10,  length_includes_head=True, head_width=x_max/50, head_length=y.max()/50, fc='k', ec='k').set_color('red')
            #axes[i,j].set_xlim(left=dframe['NORESTE']['PM2.5'].min(),
            # right=dframe['NORESTE']['PM2.5'].max())
        k+=1

fig.text(0.5, 0.04, 'PM 2.5 Concentration (ppm)', ha='center', va='center')
fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')
#fig.suptitle(station)
#print(station)

# 0,12,11,9,8,3,2,1

#%% 
fig.savefig('./Monterrey/images/histPM25_allstations2.png')

#%% mean concentration limit
dframe2=dframe.swaplevel('ESTACION','MEDIDA',axis=1).loc[:,'PM2.5']

#Assumptions validation The mean remains constant over the period of time

#qq-plot of log normal distribution of the pollutants
import statsmodels.api as sm
import numpy as np
dframe.swaplevel('ESTACION','MEDIDA',axis=1).loc[:,'PM2.5'].unstack('ESTACION').reset_index().rename(columns={0:'value'}).to_csv('AMMPM2_5long.csv')

#Use the area under the curve to compare the proportions of passings in PM2.5

#Do a reidual plot to justify that the mean doesn`t change significantly over the years.


#%%
