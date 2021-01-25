import numpy as np
import matplotlib.pyplot as plt
import h5py
with h5py.File('/home/dz2336/Desktop/Research/Project/hdp/data/real/Achilles_10252013_eeg_theta_1.mat','r') as f:
    tlfp = (f['TLFP'][()]);
    tph = f['Tph'][()];

with h5py.File('/home/dz2336/Desktop/Research/Project/hdp/data/real/Achilles_10252013_sessInfo.mat', 'r') as f:
    ## load spike info
    spikes_times = np.array(f['sessInfo']['Spikes']['SpikeTimes'])[0];
    spikes_cells = np.array(f['sessInfo']['Spikes']['SpikeIDs'])[0];
    pyr_cells = np.array(f['sessInfo']['Spikes']['PyrIDs'])[0];
    
    ## load location info ## all in maze
    locations_2d = np.array(f['sessInfo']['Position']['TwoDLocation']).T;
    locations = np.array(f['sessInfo']['Position']['OneDLocation'])[0];
    locations_times = np.array(f['sessInfo']['Position']['TimeStamps'])[:,0];
    linspeed_raw = np.array(f['sessInfo']['Position']['linspeed_raw'])[0];
    linspeed_sm = np.array(f['sessInfo']['Position']['linspeed_sm'])[0];
    
    ## load maze epoch range
    maze_epoch = np.array(f['sessInfo']['Epochs']['MazeEpoch'])[:,0];
    wake_epoch = np.array(f['sessInfo']['Epochs']['Wake']);
    
time_in_maze = ((spikes_times >= maze_epoch[0])*(spikes_times <= maze_epoch[1]));

spikes_times = spikes_times[time_in_maze];
spikes_cells = spikes_cells[time_in_maze];

cell_mask = np.isin(spikes_cells, pyr_cells);
spikes_times = spikes_times[cell_mask];
spikes_cells = spikes_cells[cell_mask];

bin_size = 25; ## change bin size to whatever you want here

binned_spike_times = np.array(np.floor(spikes_times*1000/bin_size), dtype='int');
spike_by_neuron = np.zeros((binned_spike_times.max() - binned_spike_times.min()+1, pyr_cells.shape[0]));

cell_dic = {};
for i,v in enumerate(pyr_cells):
    cell_dic[int(v)] = i;
    
for it in range(binned_spike_times.shape[0]):
    spike_by_neuron[binned_spike_times[it]-binned_spike_times.min(), cell_dic[spikes_cells[it]]] += 1;
    
tph_binned_time = np.array(np.floor((np.arange(binned_spike_times.min(),binned_spike_times.max()+1)*bin_size/1000)*1250), dtype='int');
tph_vec = tph[0][tph_binned_time]
tlpf_vec = tlfp[0][tph_binned_time]

binned_locations_times = np.array(np.floor(locations_times*1000/bin_size), dtype='int');
non_na = (~np.isnan(linspeed_raw));
binned_locations_times = binned_locations_times[non_na];
locations = locations[non_na];
linspeed_raw = linspeed_raw[non_na];
linspeed_sm = linspeed_sm[non_na];
#locations_2d = locations_2d[non_na];

locations_vec = np.zeros(spike_by_neuron.shape[0])+np.nan;
linspeed_vec = np.zeros(spike_by_neuron.shape[0])+np.nan;

for it in range(len(binned_locations_times)):
    locations_vec[binned_locations_times[it] - binned_spike_times.min()] = locations[it];
    linspeed_vec[binned_locations_times[it] - binned_spike_times.min()] = linspeed_raw[it];
    
spike_by_neuron_use = spike_by_neuron[~np.isnan(locations_vec)];
locations_vec = locations_vec[~np.isnan(locations_vec)];
tph_vec = tph_vec[~np.isnan(linspeed_vec)]
tlpf_vec = tlpf_vec[~np.isnan(linspeed_vec)]
linspeed_vec = linspeed_vec[~np.isnan(linspeed_vec)];


def rolling_max(x):
    idx_split = [];
    
    for idx, val in enumerate(x):
        if val < 0.025 or val > 1.575:
            tmp = x[max(idx-20,0):min(idx+20,len(x))];
            if val == min(tmp) or val==max(tmp):
                idx_split.append(idx)
    idx_split = np.array(idx_split)
    return idx_split

idx_split = rolling_max(locations_vec)
idx_split = np.delete(idx_split, np.where(np.abs(np.diff(locations_vec[idx_split])) < 1)[0])

from matplotlib.ticker import FormatStrFormatter
## check if you have removed all the stop stages

fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot(111);
fsz = 14;
ll = 4000
plt.plot(locations_vec[:ll])
ax1.set_xlabel('Time (s)',fontsize=fsz,fontweight='normal');
ax1.set_ylabel('Position (m)',fontsize=fsz,fontweight='normal');
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.setp(ax1.get_xticklabels(), fontsize=fsz);
plt.setp(ax1.get_yticklabels(), fontsize=fsz);
ax1.set_xticks((0,2000,4000,6000));
ax1.set_xticklabels((0,50,100,150));
#ax1.set_xticks((0,2000,4000,6000,8000,10000))
#ax1.set_xticklabels((0,50,100,150,200,250))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.scatter(idx_split[idx_split<ll], locations_vec[idx_split[idx_split<ll]], c='red')

## save data
sio.savemat("../data/achilles_data/Achilles_data.mat",
            {'trial':idx_split, 'spikes':spike_by_neuron_use, 'loc':locations_vec, 'lfp':tlpf_vec})


