# run this script using the following bash commands:
'''
for i in {1..12}
do
   echo $i
   nohup python pi-vae_rat_data_try_different_random_seed.py $i &
done
'''

import numpy as np
import scipy.stats as ss
import scipy.special as ssp
import sys
import scipy.io as sio
sys.path.append("../code/")
from pi_vae import *
from util import *
from keras.callbacks import ModelCheckpoint

seed_vec = [111,222,333,444,555,666,777,888,999,1000,1100,1200];
seed_use = seed_vec[int(sys.argv[1])-1];

## load data
rat_data = sio.loadmat("../data/achilles_data/Achilles_data.mat")
## load trial information
idx_split = rat_data['trial'][0]
## load spike data
spike_by_neuron_use = rat_data['spikes']
## load locations
locations_vec = rat_data['loc'][0]
## load lfp
tlfp_vec = rat_data['lfp'][0]

u_all = np.array(np.array_split(np.hstack((locations_vec.reshape(-1,1),np.zeros((locations_vec.shape[0],2)))), idx_split[1:-1], axis=0))
x_all = np.array(np.array_split(spike_by_neuron_use, idx_split[1:-1], axis=0))
for ii in range(len(u_all)):
    u_all[ii][:,int(ii%2)+1] = 1;
    
trial_ls = np.arange(len(u_all));
np.random.seed(666);
random_ls = np.random.permutation(trial_ls);

u_train = u_all[trial_ls[:68]];
x_train = x_all[trial_ls[:68]];

u_valid = u_all[trial_ls[68:76]];
x_valid = x_all[trial_ls[68:76]];

u_test = u_all[trial_ls[76:]];
x_test = x_all[trial_ls[76:]];

## fit pca
from sklearn.decomposition import PCA
pca_raw = PCA(n_components=2);
pca_raw_rlt = pca_raw.fit_transform(np.concatenate(x_all));

## fit pi-vae
np.random.seed(seed_use);
vae = vae_mdl(dim_x=x_all[0].shape[-1], 
                   dim_z=2,
                   dim_u=u_all[0].shape[-1], 
                   gen_nodes=60, n_blk=2, mdl='poisson', disc=False, learning_rate=5e-4)

model_chk_path = '../results/rat_2d_'+str(seed_use)+'_pivae.h5' ##999, 777
mcp = ModelCheckpoint(model_chk_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
s_n = vae.fit_generator(custom_data_generator(x_train, u_train),
              steps_per_epoch=len(x_train), epochs=600, 
              verbose=1,
              validation_data = custom_data_generator(x_valid, u_valid),
              validation_steps = len(x_valid), callbacks=[mcp]);

vae.load_weights(model_chk_path);
np.random.seed(666)
tf.random.set_random_seed(666)
elbo_samples = np.zeros(100)
for sample in range(len(elbo_samples)):
    elbo_samples[sample] = vae.evaluate_generator(custom_data_generator(x_all[:76], u_all[:76]), steps = len(x_all[:76]))

np.save('../results/rat_2d_'+str(seed_use)+'_pivae_elbo_samples.npy', elbo_samples)
