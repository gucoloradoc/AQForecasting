#%% Libraries needed for model formulation

#pylint: disable=no-name-in-module
from tensorflow import keras
import os
import numpy as np
import pandas as pd



#%% Retrieving the data
#dframe=pd.read_pickle('./AMMfull.pkl')
dframe=pd.read_csv("Monterrey/data/imputed/data/NOROESTE.csv", 
    parse_dates=["FECHA"], infer_datetime_format=True).set_index("FECHA")
#%% Preparation of the data.
#dframe.mean(axis=0).unstack('ESTACION')
#df2=dframe['NOROESTE'].fillna(method='ffill').as_matrix()
df2=dframe.values
#Max normalization
df2=df2/np.nanmax(df2)
#%% Looking back lookback, every step, we will predict the 
# concentration in a delay.
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
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
#%%
lookback = 72
step = 1
delay = 24
batch_size = 128

train_gen = generator(df2,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=35000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
val_gen = generator(df2,
                    lookback=lookback,
                    delay=delay,
                    min_index=35001,
                    max_index=40000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(df2,
                     lookback=lookback,
                     delay=delay,
                     min_index=40001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

#%%

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (40000 - 35001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(df2) - 40001 - lookback) // batch_size

#%% Naieve Method
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    
evaluate_naive_method()

#%% Basic ANN Model
#pylint: disable=import-error
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, df2.shape[-1])))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mean_squared_error'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=20,
                              epochs=32,
                              validation_data=val_gen,
                              validation_steps=val_steps)

#%% Training metrics
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#%% Test metrics

