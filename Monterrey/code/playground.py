#%% Plotting playground
import matplotlib.pyplot as plt

#pylint: disable=undefined-variable
fig, axes= plt.subplots(5,3)
stations=dframe.columns.levels[0].to_list()
measures=dframe.columns.levels[1].to_list()
#%%
k=0
for i in range(0,5):
    for j in range(0,3):
        axes[i,j].plot(dframe.loc[:,stations[0]].iloc[:,k], 
            label=measures[k])
        axes[i,j].legend(loc='upper right')
        k+=1

fig.suptitle(stations[0])
fig

#%%
dframe[dframe.index.weekday== 5]['CENTRO']['NOX'].plot()
dframe[dframe.index.hour== 17]['CENTRO']['NOX'].hist(bins=50)


#%%
