#%% Libraries needed for model formulation

#pylint: disable=no-name-in-module
from tensorflow import keras
from sklearn import metrics
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
import shutil
#%% creating folder to save outputs
out_path="Monterrey/ANN_output/"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
os.makedirs(out_path)
sys.stdout = open(out_path+'/console_output.txt', 'w')
shutil.copy2('Monterrey/code/new_generator.py', out_path)
#%% Retrieving the data
#dframe=pd.read_pickle('./AMMfull.pkl')
dframe=pd.read_csv("Monterrey/data/imputed/data/NOROESTE.csv", 
    parse_dates=["FECHA"], infer_datetime_format=True).set_index("FECHA")

station="NOROESTE"
pollutants=list(dframe.columns)
#%% Preparation of the data, normalization
#dframe.mean(axis=0).unstack('ESTACION')
#df2=dframe['NOROESTE'].fillna(method='ffill').as_matrix()
dframe_norm=dframe.copy()
#df2=dframe.values
norm_guide={'CO':('max', 0.1),
    'NO': ('max', 0.1),
    'NO2': ('max', 0.1),
    'NOX': ('max', 0.1),
    'O3': ('max', 0.01),
    'PM10': ('max', 0.1),
    'PM2.5': ('max', 0.1),
    'PRS': ('max', 0),
    'RAINF': ('max', 0.001),
    'RH': ('max', 0),
    'SO2': ('max', 0.1),
    'SR': ('max', 0),
    'TOUT': ('max',0),
    'WDR': ('max', 0),
    'WSR': ('max', 0.1),
    }

norm_param=dict.fromkeys(norm_guide.keys())

for p in pollutants:
    if norm_guide[p][0]=='log':
        dframe_norm[p]=np.log(dframe[p]+norm_guide[p][1])
    elif norm_guide[p][0]=='mean':
        dframe_norm[p]=(dframe[p]-dframe[p].mean())/dframe[p].std()
        norm_param[p]=(dframe[p].mean(),dframe[p].std())
    elif norm_guide[p][0]=='max':
        dframe_norm[p]=(dframe[p]-dframe[p].min())/(dframe[p].max()-dframe[p].min())
        norm_param[p]=(dframe[p].min(),dframe[p].max())
    if norm_guide[p][0]=='none':
        dframe_norm[p]=dframe[p]+norm_guide[p][1]


df2=dframe_norm.resample('4H').mean().values

#%% Generator over all the dataset
def generator_all(data, predictors, lookback, delay, batch_size=128, step=1, target=5):
    """ Generator form the sequence data. It will generate the observations to feed the neural network.
    From all the available data we will create the training, validation and test set. Data is expected 
    to be a numpy array, predictor a scalar or array of column idexes and all the other scalars.
    """
    max_index= len(data) - delay - 1
    #With no overlaping in the observations:
    len_obs=len(data)//(lookback//step)
    samples=np.zeros((len_obs, lookback//step, len(predictors)))
    targets=np.zeros((len_obs)) #Dont't know yet if it is needed the last dimension
    min_index=lookback//step

    for i in range(0, len_obs):
        indices=range(min_index-lookback, min_index,step)
        samples[i]=data[indices][:,predictors]
        targets[i]=data[min_index,target]
        min_index=min(min_index+lookback,max_index)
    return samples, targets


#%% Generator testing
#import numpy as np
#test_array=np.transpose(np.array([[1,2,3,4,5,6,7,8,9,10],
#                     [21,22,23,24,25,26,27,28,29,30]]))
#                     [11,12,13,14,15,16,17,18,19,20],

#gen_in, gen_out=generator_all(test_array,[0,1,2],2,1,target=2)

#%% Data set from generator
lookback = 6
step = 1
delay = 6
batch_size = 32
predictors=[5,9,11,12,13,14]
target=5 #PM10 (5), check the order in dframe

train_percent=0.7
val_percent=0.15
test_percent=0.15

obs_len=len(df2)

gen_in, gen_out=generator_all(dframe_norm.values,predictors,lookback,delay,target=target,step=step)

#%% Training, validation and test partition
train_percent=0.7
val_percent=0.15
#test_percent=0.15

from random import sample, seed
seed(10)
rand_index=sample(range(0, len(gen_in)), len(gen_in))

train_max_ind=int(train_percent*len(gen_in))
val_max_ind=int(val_percent*len(gen_in)+train_max_ind)
#test_max_ind=int(len(gen_in)-val_max_ind)

train_set = gen_in[rand_index[:train_max_ind]]
train_tar = gen_out[rand_index[:train_max_ind]]

val_set = gen_in[rand_index[(train_max_ind+1):val_max_ind]]
val_tar = gen_out[rand_index[(train_max_ind+1):val_max_ind]]

test_set = gen_in[rand_index[val_max_ind:]]
test_tar = gen_out[rand_index[val_max_ind:]]

#%% Basic ANN Model
#pylint: disable=import-error
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam

#accuracy metric for regression chosen: r2, RMSE
from tensorflow.keras import backend as K #Required for tensorflow math

def coeff_determination(y_true, y_pred):
    """ Coefficient of determination r_squared, with the keras backend.
    """
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def RMSE_PM(y_true, y_pred,normalization=norm_guide["PM10"][0],params=norm_param["PM10"]):
    """ Computes RSE with the keras backend, according to the normalization used.
    """

    if normalization=='log':
        RSE=K.sqrt(K.sum(K.square(K.exp(y_pred)-K.exp(y_true))))
    elif normalization=='max':
        y_pred=y_pred*(params[1]-params[0])+params[0]
        y_true=y_true*(params[1]-params[0])+params[0]
        RSE=K.sqrt(K.sum(K.square(y_pred-y_true)))
    else:
        RSE=K.sqrt(K.sum(K.square(y_pred-y_true)))
    return RSE
#%% ANN Model definition
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, len(predictors))))
model.add(layers.Dense(128, activation='sigmoid', name='sigmoid'))
#model.add(layers.GRU(128, input_shape=(None, len(predictors)),
#                    dropout=0.5,
#                    recurrent_dropout=0.5))
#model.add(layers.Dense(32, activation='tanh'))
model.add(layers.Dense(32, activation='linear', name='linear'))
#model.add(layers.Dense(128, activation='relu', name='relu_1'))
#model.add(layers.Dense(64, activation='relu', name='relu_2'))
#model.add(layers.Dense(32, activation='relu', name='relu_3'))
#model.add(layers.Dense(32, activation='relu', name='relu_4'))
#model.add(layers.Dense(32, activation='relu', name='relu_5'))
#model.add(layers.Dense(32, activation='relu', name='relu_6'))
#model.add(layers.Dense(64, activation='relu', name='relu_7'))
#model.add(layers.Dense(128, activation='relu', name='relu_8'))
model.add(layers.Dense(1))

#%% ANN model compilation
sys.stdout = open(out_path+'/model_training_status.txt', 'w')
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[coeff_determination, RMSE_PM])


#%% ANN model fitting

history = model.fit(train_set,train_tar, 
            epochs=100,
            validation_data=(val_set, val_tar))

#%% Saving the model 
model.save(out_path+"/PM10_NOROESTE_Predictor_"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+".h5")
from tensorflow.keras.utils import plot_model
plot_model(model, to_file=(out_path+'/model.png'), show_shapes=True)

#%% Training metrics
sys.stdout = open(out_path+'/training_metrics.txt', 'w')

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

acc=history.history['coeff_determination']
val_acc=history.history['val_coeff_determination']

train_rmse=history.history['RMSE_PM']/np.sqrt(len(test_set))
val_rmse=history.history['val_RMSE_PM']/np.sqrt(len(val_set))
epochs = range(len(loss))

plt.figure()

plt.subplot(311)
plt.plot(epochs, np.log(loss), 'bo', alpha=0.5, label='Training loss')
plt.plot(epochs, np.log(val_loss), 'b', label='Validation loss')
#plt.plot(epochs, np.ones(len(epochs))*val_naive_loss, color='orange')
#plt.plot(epochs, np.ones(len(epochs))*train_naive_loss, color='red')
plt.title('Training and validation loss and accuracy ($r^2$)')
plt.legend()

plt.subplot(312)
plt.plot(epochs, acc, 'bo', alpha=0.5, label='Training $r^2$')
plt.plot(epochs, val_acc, 'b', label='Validation $r^2$')
#plt.plot(epochs, np.ones(len(epochs))*val_naive_r2, color='orange')
#plt.plot(epochs, np.ones(len(epochs))*train_naive_r2, color='red')
plt.ylim([-1,1])
plt.legend()

plt.subplot(313)
plt.plot(epochs, train_rmse, 'bo', alpha=0.5, label='Training $RMSE$')
plt.plot(epochs, val_rmse, 'b', label='Validation $RMSE$')
plt.legend()
plt.savefig(out_path+"/Train_val_loss_acc.png", dpi=300)
#plt.show()

#%% Test metrics############################
print('Running test metrics')

#Evaluation over the test set

test_results=model.evaluate(test_set, test_tar)
pred_test=model.predict(test_set)

#%% sklearn metrics
test_tar.shape = (len(test_tar),1)
from model_post_processing import data_reescaling
pred_test=data_reescaling(pred_test,"PM10",norm_param, norm_guide)
test_tar=data_reescaling(test_tar,"PM10",norm_param, norm_guide)
det_coeff= metrics.r2_score(test_tar,pred_test)
print("r2_secore: "+str(det_coeff))
my_r2_score=coeff_determination(K.variable((test_tar)),K.variable((pred_test)))
det_coeff_v2=K.eval(my_r2_score)
print("my r2_secore: "+str(det_coeff_v2))
rmse_test= np.sqrt(metrics.mean_squared_error(np.exp(test_tar),np.exp(pred_test)))
print("RMSE: "+ str(rmse_test))
my_RMSE_test=RMSE_PM(K.variable((test_tar)),K.variable((pred_test)))
rmse_test_v2=K.eval(my_RMSE_test)/np.sqrt(len(pred_test))
print("My RMSE: "+ str(rmse_test_v2))

#Coefficient of determination (r^2)
plt.figure()
plt.plot(pred_test, test_tar,'o', alpha=0.05)
plt.plot([min(min(pred_test),min(test_tar)),
        min(max(pred_test),max(test_tar))],
        [min(min(pred_test),min(test_tar)),
        min(max(pred_test),max(test_tar))],
        'r--')
plt.annotate(("$R^2= $"+str(round(det_coeff,3))),(min(max(pred_test),max(test_tar))*(3/4),
            min(max(pred_test),max(test_tar))*(4/4)))
plt.ylabel("Measured")
plt.xlabel("Predicted")
plt.savefig(out_path+"/R2_test.png", dpi=300)


#%%
