import numpy as np
import scipy.io
import os
import csv
import torch
import matplotlib.pyplot as plt




def normalize_int16(x):
    x = x.astype(np.float32)
    mu = np.mean(x)
    sd = np.std(x)
    x -= mu
    x /= sd
    return x


class SpikeDataLoader(object):
    def __init__(self, file_dirs, file_names, batch_size, shuffle, clusters2remove=None, transform=None,dl_size=-1,dl_offset=1):
        """
        Args:

        """
        self.N_CHANNELS_OUT = 9
        self.N_SAMPLES = 32
        self.shuffle = shuffle

        self._index = 0
        self.batch_size = batch_size
        if clusters2remove is None:
            clusters2remove = [0, 1]
        self.file_dirs = file_dirs
        self.spk_file_names = []
        self.spk_data = np.empty((self.N_CHANNELS_OUT,self.N_SAMPLES,0)).astype(np.float32)
        self.clu_files_arr = np.empty((0,1))
        self.clu_files_arr_idx = np.empty((0))
        self.file_switch = np.zeros((len(file_names)), dtype=int)
        self.n_channels = []
        for i_file in range(len(file_names)):
            # mange file names
            clu_file = file_names[i_file].replace('.spk.', '.clu.')
            if not clu_file.endswith(".mat"):
                clu_file += ".mat"
            spk_file = file_names[i_file].replace('.clu.', '.spk.').replace(".mat", "")
            # get the number of channels in shank
            shank_idx_arr, n_channels_arr = read_channels(os.path.join(file_dirs[i_file], "channels.csv"))
            shank = int(spk_file.split(".")[-1])
            n_channels = n_channels_arr[shank_idx_arr.index(shank)]
            self.n_channels.append(n_channels)
            # read spikes
            self.spk_file_names.append(spk_file)
            spk_data = normalize_int16(read_spk(os.path.join(file_dirs[i_file], spk_file),n_channels))
            spk_data = spk_data[n_channels - self.N_CHANNELS_OUT:,:,:]
            # load clu files
            curr_clu = read_clu(os.path.join(file_dirs[i_file], clu_file))
            curr_clu_idx = np.arange(curr_clu.size)
            # remove clusters with noise spikes
            if clusters2remove:
                for i_cluster in range(len(clusters2remove)):
                    curr_clu_idx = curr_clu_idx[(curr_clu != clusters2remove[i_cluster]).squeeze()]
                    spk_data = spk_data[:, :, (curr_clu != clusters2remove[i_cluster]).squeeze()]
                    curr_clu = curr_clu[(curr_clu != clusters2remove[i_cluster]).squeeze()]

            self.spk_data = np.concatenate((self.spk_data, spk_data), axis=2)
            self.clu_files_arr=np.concatenate((self.clu_files_arr,curr_clu),axis=0)
            self.clu_files_arr_idx=np.concatenate((self.clu_files_arr_idx,curr_clu_idx),axis=0)
            self.file_switch[i_file] = curr_clu.size
            map_file = "map" + clu_file[:-4].replace(".clu."+str(shank),".npy")
            full_map_table = np.load(os.path.join(file_dirs[i_file], map_file))
            self.good_clusters = full_map_table[full_map_table[:,1]==shank,2]
        self.spk_data = np.moveaxis(self.spk_data,-1,0)
        self.file_switch = self.file_switch.cumsum()
        self.transform = transform

        if self.shuffle:
            self.sampler = np.random.permutation(len(self.clu_files_arr))
        else:
            self.sampler = np.arange(len(self.clu_files_arr))

        if dl_offset>-1:
            self.spk_data = self.spk_data[self.sampler[dl_offset:],:,:]
            self.clu_files_arr = self.clu_files_arr[self.sampler[dl_offset:]]
            self.clu_files_arr_idx = self.clu_files_arr_idx[self.sampler[dl_offset:]]
            self.file_switch[-1] = len(self.clu_files_arr)
            if self.shuffle:
                self.sampler = np.random.permutation(len(self.clu_files_arr))
            else:
                self.sampler = np.arange(len(self.clu_files_arr))


        if dl_size > -1:
            self.spk_data = self.spk_data[self.sampler[:dl_size], :, :]
            self.clu_files_arr = self.clu_files_arr[self.sampler[:dl_size]]
            self.clu_files_arr_idx = self.clu_files_arr_idx[self.sampler[:dl_size]]
            self.file_switch[-1] = dl_size
            if self.shuffle:
                self.sampler = np.random.permutation(len(self.clu_files_arr))
            else:
                self.sampler = np.arange(len(self.clu_files_arr))


    def __len__(self):
        return self.file_switch[-1]

    def __iter__(self):
        return self

    def __next__(self):
        if self._index  + self.batch_size>=len(self):
            raise StopIteration
        spike=self.spk_data[self.sampler[self._index:self._index+self.batch_size],:,:]
        label = self.clu_files_arr[self.sampler[self._index:self._index+self.batch_size]]
        self._index += self.batch_size
        return torch.from_numpy(spike), torch.from_numpy(label)

    def reset(self):
        self._index = 0
        if self.shuffle:
            self.sampler = np.random.permutation(len(self.clu_files_arr))
        else:
            self.sampler = np.arange(len(self.clu_files_arr))

    def show_next(self):
        spike = self.spk_data[self.sampler[self._index],:,:].squeeze()
        show_spike(spike, dashed=False)

    def get_spike(self, spike_idx):
        spike = self.spk_data[spike_idx,:,:].squeeze()
        label = self.clu_files_arr[spike_idx].squeeze()
        return torch.from_numpy(spike), torch.from_numpy(label)


def show_spike(spike, dashed=False):
    spike = spike.T
    n_samples, n_chan = spike.shape
    spike += np.tile(np.arange(n_chan)*5,(n_samples, 1))
    if dashed:
        plt.plot(spike, '--')
    else:
        plt.plot(spike)
    plt.show()


def show_two_spikes(spike, spike2):
    spike = spike.T
    spike2 = spike2.T
    n_samples, n_chan = spike.shape
    spike += np.tile(np.arange(n_chan) * 5, (n_samples, 1))
    spike2 += np.tile(np.arange(n_chan) * 5, (n_samples, 1))
    plt.plot(spike)
    plt.plot(spike2, '--')
    plt.show()


def read_clu(full_file, spike_idx=-1):
    clu_data = scipy.io.loadmat(full_file)
    clu = clu_data['clu']
    clu = clu[1:]
    if spike_idx > -1:
        clu = clu[spike_idx]
    return clu


def read_spk(full_file, n_channels, n_samples=32, spike_idx=-1):
    bytes_per_sample = 2
    if spike_idx == -1:
        n_items = -1
        read_offset = 0
    else:
        n_items = n_channels * n_samples
        read_offset = spike_idx * n_items * bytes_per_sample

    with open(full_file, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.int16, count=n_items, offset=read_offset).reshape((-1, n_channels)).T
    n_spikes = raw_data.size / n_channels / n_samples
    if len(raw_data) % n_channels * n_samples > 0:
        print("err loading {}".format(full_file))
        return []
    spikes = np.reshape(raw_data, (n_channels, n_samples, int(n_spikes)), order='F')
    return spikes


def read_channels(file_name):
    file_idx = []
    n_channels = []
    with open(file_name, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            file_idx.append(int(row["shank"]))
            n_channels.append(int(row["n"]))
    return file_idx, n_channels

# clu =read_clu('D:/Drive/DL_project_OD/RealData/mF105_10/mF105_10.clu.1.mat')
# spikes = load_spk('D:/Drive/DL_project_OD/RealData/mF105_10/mF105_10.spk.1', 11, spike_idx=1)
# read_channels('D:/Drive/DL_project_OD/RealData/mF105_10/channels.csv')
# print(spikes[0,:,0])
# file_dirs = ["D:/Drive/DL_project_OD/RealData/mF105_10", "D:/Drive/DL_project_OD/RealData/mF105_10", ]
# file_clu_names = ["mF105_10.spk.1", "mF105_10.clu.2.mat", ]
#
# DS = SpikeDataSet(file_dirs, file_clu_names)
# print(DS.__getitem__(int(1)))
