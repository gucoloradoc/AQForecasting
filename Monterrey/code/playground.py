#%% Plotting playground
import matplotlib.pyplot as plt

fig, axes= plt.subplots(5,3)
statitions=dframe.columns.levels[0].to_list()
measures=dframe.columns.levels[1].to_list()
#%%
k=0
for i in range(0,5):
    for j in range(0,3):
        axes[i,j].plot(dframe.loc[:,'CENTRO'].iloc[:,k], 
            label=measures[k])
        axes[i,j].legend(loc='upper right')
        k+=1

fig

#%%
