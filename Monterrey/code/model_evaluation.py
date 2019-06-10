#%% loading libraries
#pylint: disable=no-name-in-module
from tensorflow import keras
import numpy as np
import pandas as pd

#%% loading the pretrained model

model=keras.models.load_model("modelANN_imputed_May30.h5")


#%% testing the model

test_results=model.evaluate_generator(test_gen, steps=1)

#%% Testing the generator
def data_generator(gen,steps):
    samp_imp = []
    samp_out = []
    for step in range(steps):
        samples, targets = next(gen)
        #preds = samples[:, -1, 1]
        #mae = np.mean(np.abs(preds - targets))
        #batch_maes.append(mae)
        samp_imp.append(samples)
        samp_out.append(targets)
    #print(np.mean(batch_maes))
    return samp_imp, samp_out

samp_imp, samp_out= data_generator(val_gen, val_steps)

#%%
