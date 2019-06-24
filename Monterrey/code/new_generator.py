#%%
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

#%%
