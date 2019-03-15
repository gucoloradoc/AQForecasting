#%% Data Tidying for Air quality forecasting
# Monterrey DB.
# Enter the main "Data" folder

#%%
import numpy as np
import pandas as pd
from datetime import timedelta
import os

datapath='Monterrey/data/raw/'

sheetpath='Todo 2012.xlsx'

fullsheetpath=os.path.join(datapath,sheetpath)

dframe= pd.read_excel(fullsheetpath, header=[0,1])

#%%


#%%
