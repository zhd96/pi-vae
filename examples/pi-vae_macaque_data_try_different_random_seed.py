# run this script using the following bash commands:
'''
for i in {1..12}
do
   echo $i
   nohup python pi-vae_macaque_data_try_different_random_seed.py $i &
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

# ## load macaque data

## load trial information
## starttime, endtime, number, tgtontime, gocuetime, tgtdir, tgtid
trial_dat = sio.loadmat("../data/chewie_data/Chewie_20161006_trials_array.mat")

## load spike data
dat_ = sio.loadmat('../data/chewie_data/Chewie_20161006_seq.mat');

dat_all = [[] for _ in range(8)];
tar_dir = np.unique(trial_dat['trials_array'][:,5])[:8];
trial_dat_id = np.unique(trial_dat['trials_array'][:,5],return_inverse=True)[1];

for trial_id in range(251):
    if dat_['seq'][0][trial_id]['T'][0,0] != 0:
        dat_all[trial_dat_id[trial_id]].append(dat_['seq'][0][trial_id]['y'].T[:,63:]);

dat_all = np.array([np.array(dat_all[ii]) for ii in range(8)]);


## randomly split into batches
np.random.seed(666);
trial_ls = [np.random.permutation(np.array_split(np.random.permutation(np.arange(dat_all[ii].shape[0])),24)) for ii in range(8)];

x_all = [];
u_all = [];
for ii in range(24): # 24 batches
    x_tr = [];
    u_tr = [];
    for jj in range(8): # 8 different directions
        x_tmp = np.concatenate(dat_all[jj][trial_ls[jj][ii]])#[:,:-1];
        u_tmp = np.ones((x_tmp.shape[0],1))*jj;
        x_tr.append(x_tmp);
        u_tr.append(u_tmp);
    x_all.append(np.concatenate(x_tr));
    u_all.append(np.concatenate(u_tr));

x_all = np.array(x_all);
u_all = np.array(u_all);

x_train = x_all[:20];
u_train = u_all[:20];

x_valid = x_all[20:22];
u_valid = u_all[20:22];

x_test = x_all[22:];
u_test = u_all[22:];

## check pca results
from sklearn.decomposition import PCA
pca_raw = PCA(n_components=x_all[0].shape[-1]);
pca_raw_rlt = pca_raw.fit_transform(np.concatenate(x_all));

# ## fit pi-vae

np.random.seed(seed_use);
vae = vae_mdl(dim_x=x_all[0].shape[-1],
                   dim_z=4,
                   dim_u=np.unique(np.concatenate(u_all)).shape[0],
                   gen_nodes=60, n_blk=2, mdl='poisson', disc=True, learning_rate=5e-4)

model_chk_path = '../results/macaque_4d_'+str(seed_use)+'_pivae.h5'
mcp = ModelCheckpoint(model_chk_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
s_n = vae.fit_generator(custom_data_generator(x_train, u_train),
              steps_per_epoch=len(x_train), epochs=1000,
              verbose=1,
              validation_data = custom_data_generator(x_valid, u_valid),
              validation_steps = len(x_valid), callbacks=[mcp]);

