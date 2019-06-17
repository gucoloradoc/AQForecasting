#%% Libraries needed for model formulation

#pylint: disable=no-name-in-module
from tensorflow import keras
from sklearn import metrics
import numpy as np
import pandas as pd
from datetime import datetime
import os
#%% creating folder to save outputs
out_path="Monterrey/ANN_output/"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
os.makedirs(out_path)

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
norm_guide={'CO':('log', 0.1),
    'NO': ('log', 0.1),
    'NO2': ('log', 0.1),
    'NOX': ('log', 0.1),
    'O3': ('log', 0.01),
    'PM10': ('log', 0.1),
    'PM2.5': ('log', 0.1),
    'PRS': ('mean', 0),
    'RAINF': ('log', 0.001),
    'RH': ('mean', 0),
    'SO2': ('log', 0.1),
    'SR': ('none', 0),
    'TOUT': ('mean',0),
    'WDR': ('mean', 0),
    'WSR': ('log', 0.1),
    }

for p in pollutants:
    if norm_guide[p][0]=='log':
        dframe_norm[p]=np.log(dframe[p]+norm_guide[p][1])
    if norm_guide[p][0]=='none':
        dframe_norm[p]=dframe[p]+norm_guide[p][1]
    if norm_guide[p][0]=='mean':
        dframe_norm[p]=(dframe[p]-dframe[p].mean())/dframe[p].std()


df2=dframe_norm.values
#%% normalization

#%% Looking back lookback, every step, we will predict the 
# concentration in a delay.
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6, target=5):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            #Here my targets could be O3(4),PM10(5),PM2.5(6)
            targets[j] = data[rows[j] + delay][5]
        yield samples, targets
#%% Generators setup
lookback = 24
step = 1
delay = 24
batch_size = 64
target=5 #PM10 (5), check the order in dframe

train_gen = generator(df2,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=35000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size,
                      target=target)
val_gen = generator(df2,
                    lookback=lookback,
                    delay=delay,
                    min_index=35001,
                    max_index=40000,
                    step=step,
                    batch_size=batch_size,
                      target=target)
test_gen = generator(df2,
                     lookback=lookback,
                     delay=delay,
                     min_index=40001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size,
                      target=target)

#%% Defining number of steps
#Training steps

train_steps=50
# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (40000 - 35001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(df2) - 40001 - lookback) // batch_size

#%% Naieve Method, redefinition
def eval_naive_method(gen, steps,var):
    batch_logmse = []
    tar = []
    pred = []
    for step in range(steps):
        samples, targets = next(gen)
        preds = samples[:, -1, var] #Last index corresponds to PM10(5), PM2.5(6)
        logmse = np.log(np.mean(np.square(preds-targets)))
        #mae = np.mean(np.abs(preds - targets))
        batch_logmse.append(logmse)
        tar.extend(targets)
        pred.extend(preds)
    print("From naive assumption that the pollutant concentration"
     "\n will be the same that 24h before: \n")
    print("log(mse)= "+str(round(np.mean(batch_logmse),3)))
    from sklearn.metrics import r2_score
    print("r2= "+str(round(r2_score(tar,pred),5)))
    return round(np.mean(batch_logmse),3), round(r2_score(tar,pred),5)

print("\n Training set: \n")
train_naive_loss, train_naive_r2 = eval_naive_method(train_gen, train_steps,5)

print("\n Validation set: \n")
val_naive_loss, val_naive_r2 =eval_naive_method(val_gen, val_steps,5)

#%% Basic ANN Model
#pylint: disable=import-error
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

#accuracy metric for regression chosen: r2
from keras import backend as K #Required for tensorflow math

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#%% ANN Model definition
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, df2.shape[-1])))
model.add(layers.Dense(32, activation='sigmoid', name='sigmoid'))
#model.add(layers.Dense(256, activation='tanh'))
model.add(layers.Dense(32, activation='linear', name='linear'))
#model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='linear', name='output'))

#%% ANN model compilation
model.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mean_squared_error', coeff_determination])
history = model.fit_generator(train_gen,
                              steps_per_epoch=train_steps,
                              epochs=50,
                              validation_data=val_gen,
                              validation_steps=val_steps)

#%% Saving the model 
model.save(out_path+"/PM10_NOROESTE_Predictor_"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+".h5")
from keras.utils import plot_model
plot_model(model, to_file=(out_path+'/model.png'), show_shapes=True)

#%% Training metrics
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

acc=history.history['coeff_determination']
val_acc=history.history['val_coeff_determination']

epochs = range(len(loss))

plt.figure()

plt.subplot(211)
plt.plot(epochs, np.log(loss), 'bo', label='Training loss')
plt.plot(epochs, np.log(val_loss), 'b', label='Validation loss')
plt.plot(epochs, np.ones(len(epochs))*val_naive_loss, color='orange')
plt.plot(epochs, np.ones(len(epochs))*train_naive_loss, color='red')
plt.title('Training and validation loss and accuracy ($r^2$)')
plt.legend()

plt.subplot(212)
plt.plot(epochs, acc, 'bo', label='Training $r^2$')
plt.plot(epochs, val_acc, 'b', label='Validation $r^2$')
plt.plot(epochs, np.ones(len(epochs))*val_naive_r2, color='orange')
plt.plot(epochs, np.ones(len(epochs))*train_naive_r2, color='red')
plt.ylim([-1,1])
plt.legend()
plt.savefig(out_path+"/Train_val_loss_acc.png", dpi=300)
#plt.show()

#%% Test metrics############################
print('Running test metrics')
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

#%% sklearn metrics

det_coeff= metrics.r2_score(samp_out,pred_test)
print(det_coeff)

#Coefficient of determination (r^2)
plt.figure()
plt.plot(pred_test, samp_out,'o', alpha=0.05)
plt.plot([min(min(pred_test),min(samp_out)),
        min(max(pred_test),max(samp_out))],
        [min(min(pred_test),min(samp_out)),
        min(max(pred_test),max(samp_out))],
        'r--')
plt.annotate(("$R^2= $"+str(round(det_coeff,3))),(min(max(pred_test),max(samp_out))*(3/4),
            min(max(pred_test),max(samp_out))*(4/4)))
plt.ylabel("Measured")
plt.xlabel("Predicted")
plt.savefig(out_path+"/R2_test.png", dpi=300)

#%%
