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

    def __init__(self, filename, samples, predict, test_set, train, inference=False, normalize=False, normal_dis=False, total_dev=11, error_at5percent=False):
        """
        :param
        :param
        :param
        :return:
        """

        if train and error_at5percent:
            raise Exception("The error at 5% should be only considered for testing mode.")

        idx = range(0, total_dev)

        mat = matloader.loadmat(filename)
        self.dev_names = mat['devs']
        training_list = [i for i in idx if i != test_set[0]]
        self.input_size = samples
        self.target_size = predict
        self.normalize = normalize
        self.train = train
        self.mean = None
        self.std = None
        self.normal_dis = normal_dis
        self.min_arr = float('+inf')
        self.max_arr = float('-inf')

        if train:
            print('-> Processing following devices for training mode:')
            action_set = training_list
        else:
            print('-> Processing following devices for test mode:')
            action_set = test_set

        if self.normalize:
            _tmp = np.array([])
            for i in idx:
                _tmp = np.append(_tmp, mat['vals'][0, i][0])

            self.min_arr = min(_tmp.min(), self.min_arr)
            self.max_arr = max(_tmp.max(), self.min_arr)
            self.mean = np.mean(_tmp)
            self.std = np.std(_tmp)
            
        # Set all samples to have the number of time steps by padding 0's
        #self.dataset = np.empty((len(action_set), timesteps))
        _dataset = list()  # np.zeros((1, (samples + predict)))
        for _, val in enumerate(action_set):
            dev_name = mat['devs'][0, val][0]

            _tmp_len = len(mat['vals'][0, val][0])
            _tmp_data = mat['vals'][0, val][0]

            if self.normal_dis:
                _tmp_data = (_tmp_data-self.mean)/self.std
            else:
                _tmp_data = 2*((_tmp_data-self.min_arr) /
                               (self.max_arr-self.min_arr)) - 1

            if train:
                _dataset.append(np.array(_tmp_data).astype('float32'))
            else:
                
                if not error_at5percent:
                    _how_many = int(_tmp_len / (samples + predict))

                    # Needed to reach the end of the datasample (if not easily divisible by the input + prediction window)
                    if inference:
                        print('+ Inference mode.')
                        #_how_many += 1
                    for i in range(_how_many):
                        print("-> Processing Device({})-{:3.2f}%".format(val,
                                                                         i*100/_how_many), end='\r')
                        _one_sample = np.array(
                            _tmp_data[i*(samples+predict):(i+1)*(samples+predict)])
                        #_one_sample = np.expand_dims(_one_sample, axis=0)
                        _dataset.append(_one_sample.astype('float32'))
                else:
                    print('ELSE')
                    _one_sample = np.array(_tmp_data[_tmp_len-(samples+predict):])
                    _dataset.append(_one_sample.astype('float32'))
        self.dataset = _dataset
        pass

        #self.dataset = torch.from_numpy(np.asarray(tmp).astype(np.float32)).type(torch.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.train:
            total_data = len(self.dataset[idx])
            j = np.random.random_integers(total_data)
            if j + (self.input_size + self.target_size) <= total_data-1:
                return self.dataset[idx][j:j+self.input_size], self.dataset[idx][j+self.input_size:j+self.input_size + self.target_size]
            else:
                return self.dataset[idx][total_data - (self.input_size+self.target_size):total_data - self.target_size], self.dataset[idx][total_data - self.target_size:total_data]

        else:
            return self.dataset[idx][:self.input_size], self.dataset[idx][self.input_size:]



class NASARealTime(Dataset):
    """NASA MOSFET Thermal Overstress Aging Data Set
    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    Generates data samples
    """

    def __init__(self, filename, samples, predict, test_set, train=True, normalize=True, normal_dis=True, total_dev=11):
        """
        :param
        :param
        :param
        :return:
        """
        
        idx = range(0, total_dev)

        mat = matloader.loadmat(filename)
        self.dev_names = mat['devs']
        training_list = [i for i in idx if i != test_set[0]]
        self.input_size = samples
        self.target_size = predict
        self.normalize = normalize
        self.train = train
        self.mean = None
        self.std = None
        self.normal_dis = normal_dis
        self.min_arr = float('+inf')
        self.max_arr = float('-inf')
        self.input = []
        self.target = []

        if train:
            #print('-> Processing following devices for training mode:')
            action_set = training_list
        else:
            #print('-> Processing following devices for test mode:')
            action_set = test_set

        if self.normalize:
            _tmp = np.array([])
            for i in idx:
                _tmp = np.append(_tmp, mat['vals'][0, i][0])

            self.min_arr = min(_tmp.min(), self.min_arr)
            self.max_arr = max(_tmp.max(), self.min_arr)
            self.mean = np.mean(_tmp)
            self.std = np.std(_tmp)

        for i in action_set:
            ptr = 0
            sample = mat['vals'][0, i][0]
            dev_name = mat['devs'][0, i][0]
            if self.normalize:
                sample = (sample-self.mean)/self.std
            stop = len(sample) - self.input_size - self.target_size

            for _ in range(stop):
                input_start = ptr
                input_end = ptr + self.input_size
                target_start = input_end
                target_end = target_start + self.target_size

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

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        return self.input[idx], self.target[idx]


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