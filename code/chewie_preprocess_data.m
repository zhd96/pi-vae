%% matlab script
trials = data.trials(:,[1:3,5:7,9]);
trials_array = table2array(trials);
save('Chewie_20161006_trials_array.mat', 'trials_array');

nNeuron = 255;
nTrial = 251;

spike_array = {};
for ii = 1:nNeuron
spike_array{ii} = (data.units(ii).spikes.ts);
end
loc_array = {};
for ii = 1:nNeuron
loc_array{ii} = (data.units(ii).array);
end
save('Chewie_20161006_spike_array.mat', 'spike_array', 'loc_array');

bin_siz = 50/1000;
seq=[];
for tr=1:nTrial,
    t_start_drop = trials_array(tr,5);
    t_end_drop = trials_array(tr,2);
    bin_edge = (t_start_drop):bin_siz:(t_end_drop);
    nBin = length(bin_edge) - 1;
    spike_bin_temp = zeros(nNeuron, nBin);
    for u=1:nNeuron,
        time_use=spike_array{u};
        if isempty(time_use), continue, end;
        spike_this_temp = histc(time_use, bin_edge);
        spike_bin_temp(u,:) = spike_this_temp(1:(end-1));
    end
    seq(tr).y=spike_bin_temp;
    seq(tr).T=size(spike_bin_temp,2);
end
save('Chewie_20161006_seq.mat', 'seq');

