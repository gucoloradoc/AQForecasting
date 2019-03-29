#%% Model formulation
import keras
import os
import numpy as np
import pandas as pd


#%% Retrieving the data
dframe=pd.read_pickle('./AMMfull.pkl')
#%% Preparation of the data.
dframe.mean(axis=0).unstack('ESTACION')
df2=dframe['NOROESTE'].fillna(method='ffill').as_matrix()
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
batch_size = 1

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
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, df2.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=10,
                              epochs=10,
                              validation_data=val_gen,
                              validation_steps=val_steps)




#%%
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

#%%
