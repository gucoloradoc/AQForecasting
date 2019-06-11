#%% loading libraries
#pylint: disable=no-name-in-module
from tensorflow import keras
import numpy as np
import pandas as pd

#%% Testing the generator
def data_from_generator(gen,steps):
    samp_imp = []
    samp_out = []
    for step in range(steps):
        samples, targets = next(gen)
        #preds = samples[:, -1, 1]
        #mae = np.mean(np.abs(preds - targets))
        #batch_maes.append(mae)
        samp_imp.extend(samples)
        samp_out.extend(targets)
    #print(np.mean(batch_maes))
    return samp_imp, samp_out

samp_imp, samp_out= data_from_generator(test_gen, test_steps)

#%% loading the pretrained model

#model=keras.models.load_model("modelANN_imputed_May30.h5")


#%% testing the model

test_results=model.evaluate_generator(test_gen, steps=2)
pred_test=model.predict(np.array(samp_imp))

#Coefficient of determination (r^2)
plt.plot(pred_test, samp_out,'o', alpha=0.05)
plt.plot([min(min(pred_test),min(samp_out)),
        min(max(pred_test),max(samp_out))],
        [min(min(pred_test),min(samp_out)),
        min(max(pred_test),max(samp_out))],
        'r--')
#%% sklearn metrics
from sklearn import metrics
det_coeff= metrics.r2_score(samp_out,pred_test)
print(det_coeff)
#%%


#%%
