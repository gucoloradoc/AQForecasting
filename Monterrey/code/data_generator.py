"""Script to generate the dataset of observations from air quality measures.
"""
def dataset_generator(data, lookback, delay, batch_size=64, 
                        step=1, target=5):
#%% Generator redefinition
def generator(data, predictors, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=1, target=5):
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
                           len(predictors)))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            #print(rows)
            #print(samples.shape)
            indices = range(rows[j] - lookback, rows[j], step)
            #print(indices)
            samples[j] = data[indices][:,predictors]
            #Here my targets could be O3(4),PM10(5),PM2.5(6)
            targets[j] = data[rows[j] + delay][target]
        yield samples, targets

#%% Generators setup

lookback = 24
step = 4
delay = 24
batch_size = 32
predictors=[5,9,11,12,13,14]
target=5 #PM10 (5), check the order in dframe

train_percent=0.7
val_percent=0.15
test_percent=0.15

obs_len=len(df2)

train_max_ind=int(train_percent*obs_len)
val_max_ind=int(val_percent*obs_len+train_max_ind)
test_max_ind=int(test_percent*obs_len+val_max_ind)

train_gen = generator(df2,
                      lookback=lookback,
                      delay=delay,
                      predictors=predictors,
                      min_index=0,
                      max_index=train_max_ind,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size,
                      target=target)
val_gen = generator(df2,
                    predictors=predictors,
                    lookback=lookback,
                    delay=delay,
                    min_index=train_max_ind+1,
                    max_index=val_max_ind,
                    step=step,
                    batch_size=batch_size,
                      target=target)
test_gen = generator(df2,
                     predictors=predictors,
                     lookback=lookback,
                     delay=delay,
                     min_index=val_max_ind+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size,
                      target=target)


#%%
