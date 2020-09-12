import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as matloader

_DISCRIPTION = '''
The file dR15Devs has following devices:
idx    Device       IoT device idx

00 -> 'Dev#11'
01 -> 'Dev#12'
02 -> 'Dev#14'
03 -> 'Dev#24'
04 -> 'Dev#25'
05 -> 'Dev#26'
06 -> 'Dev#29'
07 -> 'Dev#32'
08 -> 'Dev#33'
09 -> 'Dev#35'
10 -> 'Dev#36'      'Dev#4'
11 -> 'Dev#37'
12 -> 'Dev#38'
13 -> 'Dev#8'
14 -> 'Dev#9'
It also contains two variables:
    a. devs: the name of devices,
    b. vals: the dR of each dev
'''


class NASADataSet(Dataset):
    """NASA MOSFET Thermal Overstress Aging Data Set
    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

    Generates data samples

    """

    def __init__(self, filename, input_size, predict_size, test_set, train, focus, total_dev=11):
        """
        :param
        :param
        :param
        :return:
        """

        self.idx = range(0, total_dev)
        self.ptr = np.zeros(total_dev - 1, dtype=int)
        self.how_many = np.zeros(total_dev - 1, dtype=int)
        self.mat = matloader.loadmat(filename)
        self.dev_names = self.mat['devs']
        training_list = [i for i in self.idx if i != test_set[0]]
        self.input_size = input_size
        self.predict_size = predict_size
        self.train = train
        self.mean, self.std = self.normalize()
        self.error_index = np.zeros(total_dev - 1, dtype=int)
        self.focus = focus

        if train:
            action_set = training_list
            print('-> Processing following devices for training mode: {}'.format(action_set))
            
        else:
            action_set = test_set
            print('-> Processing following devices for test mode: {}'.format(action_set))
            
        _dataset = list()
        for i, val in enumerate(action_set):
            dev_name = self.mat['devs'][0, val][0]

            _tmp_data = self.mat['vals'][0, val][0]
            _tmp_data = (_tmp_data-self.mean)/self.std

            _tmp_len = len(self.mat['vals'][0, val][0])
            pad = _tmp_len % (input_size + predict_size)

            _tmp_data = np.pad(_tmp_data, (0, pad), 'constant', constant_values=(0, _tmp_data[-1]))
            _how_many = int(_tmp_len / (input_size + predict_size))
            self.how_many[i] = _how_many

            if train:
                _dataset.append(np.array(_tmp_data).astype('float32'))
                self.error_index[i] = self.find_index(_tmp_data >= 0.02)
                if self.error_index[i] == -1:
                    self.error_index[i] = total_dev_len - 1
                
                self.error_index[i] = self.error_index[i] // (self.predict_size + self.input_size)
                self.error_index[i] -= 1
                if self.error_index[i] < 0:
                    self.error_index[i] = 0
            else:
                for j in range(_how_many):
                    print("-> Processing Device({})-{:3.2f}%".format(val, j*100/_how_many), end='\r')
                    _one_sample = np.array(_tmp_data[j*(input_size+predict_size):(j+1)*(input_size+predict_size)])

                    _dataset.append(_one_sample.astype('float32'))
             
        self.dataset = _dataset

    def normalize(self):
        self._tmp = np.array([])
        for i in self.idx:
            self._tmp = np.append(self._tmp, self.mat['vals'][0, i][0])
        mean = np.mean(self._tmp)
        std = np.std(self._tmp)
        return mean, std

    def find_index(self, arr):
        """Find the threshold index, if it exists, and return it.
        arr is a boolean array comparing each element >= threshold value,
        where True if it is >=, and False if not"""
        print('-> Finding threshold index...')
        # Determine if threshold value exists in current array
        # any() returns True if any value of the array is True and False otherwise
        if not arr.any():
            return -1
        
        # Return the FIRST index that is >= the threshold value (True), which is 0.05 as from the NASA paper.
        return np.argmax(arr)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.train:
            if self.ptr[idx] == self.how_many[idx]:
                if self.focus:
                    print('Looping device {} back to {}'.format(idx, self.error_index[idx]))
                    self.ptr[idx] = self.error_index[idx]
                else:
                    print('\nLooping device {} back to 0\n'.format(idx))
                    self.ptr[idx] = 0
            input_start = self.ptr[idx] * (self.input_size + self.predict_size)
            input_stop = input_start + self.input_size
            
            target_start = input_stop
            target_stop = target_start + self.predict_size

            if idx == len(self.dataset) - 1:
                self.ptr += 1

            return self.dataset[idx][input_start:input_stop], self.dataset[idx][target_start:target_stop]

        else:
            return self.dataset[idx][:self.input_size], self.dataset[idx][self.input_size:]



class NASARealTime(Dataset):
    """NASA MOSFET Thermal Overstress Aging Data Set
    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    Generates data samples
    """

    def __init__(self, filename, input_size, predict_size, test_set, train=True, focus=None, total_dev=11):
        """
        :param
        :param
        :param
        :return:
        """
        
        self.idx = range(0, total_dev)
        self.ptr = np.zeros(total_dev - 1, dtype=int)
        self.how_many = np.zeros(total_dev - 1, dtype=int)
        self.mat = matloader.loadmat(filename)
        self.dev_names = self.mat['devs']
        training_list = [i for i in self.idx if i != test_set[0]]
        self.input_size = input_size
        self.predict_size = predict_size

        self.train = train
        self.mean, self.std = self.normalize()

        self.input = []
        self.target = []

        if train:
            action_set = training_list
            print('-> Processing following devices for training mode: {}'.format(action_set))
            
        else:
            action_set = test_set
            print('-> Processing following devices for test mode: {}'.format(action_set))

        for i in action_set:
            ptr = 0
            sample = self.mat['vals'][0, i][0]
            dev_name = self.mat['devs'][0, i][0]
  
            stop = len(sample) - self.input_size - self.predict_size + 1

            for _ in range(stop):
                input_start = ptr
                input_end = ptr + self.input_size
                target_start = input_end
                target_end = target_start + self.predict_size

                self.input.append(sample[input_start:input_end])
                self.target.append(sample[target_start:target_end])
                ptr += 1

        self.input = np.array(self.input)
        self.input = torch.from_numpy(self.input)
        self.input = self.input.type(torch.FloatTensor)
        #print(self.input.shape)
        self.target = np.array(self.target)

        self.target = torch.from_numpy(self.target)
        self.target = self.target.type(torch.FloatTensor)

    def normalize(self):
        self._tmp = np.array([])
        for i in self.idx:
            self._tmp = np.append(self._tmp, self.mat['vals'][0, i][0])
        mean = np.mean(self._tmp)
        std = np.std(self._tmp)
        return mean, std

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]


    # def __len__(self):
    #     return len(self.input)

    # def __getitem__(self, idx):

    #     return self.input[idx], self.target[idx]


if __name__ == "__main__":
    i_size = 20
    t_size = 104
    b_size = 4
    e_size = 1000
    dt = NASADataSet('./utility/dR15Devs.mat', i_size, t_size, [1], True, True, True)
    dl = DataLoader(dt, b_size, True, num_workers=1, drop_last=True)
    for e in range(e_size):
        for _, (x, t) in enumerate(dl):
            for b_i in range(b_size):
                assert len(x[b_i]) == i_size
                assert len(t[b_i]) == t_size
    pass