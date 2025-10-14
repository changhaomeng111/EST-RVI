import numpy as np
import pandas as pd
from tqdm import tqdm


class DataProcessor:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def max_min_normalization(x, max_val, min_val):
        return (x - min_val) / (max_val - min_val)

    def load_traffic_data(self):
        Pr_Hi = np.load(f'{self.config.data_path}Pr_Hi.npy')
        Pf_E = np.load(f'{self.config.data_path}Pf_E.npy')
        adj = np.load(f'{self.config.data_path}adj_matrix.npy')

        return Pr_Hi, Pf_E, adj

    def load_event_data(self):
        """Load event data"""
        datatxt = pd.read_table(f"{self.config.data_path}Crowd_queries.dat",
                                delimiter=",", header=None)
        datatxt.columns = ["time", "wayid", 'query_count', 'visibility', 'precip_accum']
        return datatxt

    def preprocess_data(self, data, n_data, times_inp,
                        spatial_inp, values_inp, varis_inp, time_len,
                        train_rate, seq_len, pre_len):
        """Data preprocessing"""

        # First expand dimensions, then concatenate
        a_expanded = data[:, :, np.newaxis]
        b_expanded = n_data[:, :, np.newaxis]
        data = np.concatenate([np.array(a_expanded), np.array(b_expanded)], axis=2)

        train_size = int(time_len * train_rate)
        train_data = data[0:train_size]
        test_data = data[train_size:time_len]

        # Process event data
        spatial_inp_train = spatial_inp[0:train_size]
        spatial_inp_test = spatial_inp[train_size:time_len]
        times_inp_train = times_inp[0:train_size]
        times_inp_test = times_inp[train_size:time_len]
        varis_inp_train = varis_inp[0:train_size]
        varis_inp_test = varis_inp[train_size:time_len]

        values_inp_max = np.max(np.max(values_inp))
        values_inp = values_inp / values_inp_max
        values_inp_train = values_inp[0:train_size]
        values_inp_test = values_inp[train_size:time_len]

        # Expand dimensions
        spatial_inp_train = np.repeat(spatial_inp_train[..., np.newaxis], 2, axis=-1)
        spatial_inp_test = np.repeat(spatial_inp_test[..., np.newaxis], 2, axis=-1)
        times_inp_train = np.repeat(times_inp_train[..., np.newaxis], 2, axis=-1)
        times_inp_test = np.repeat(times_inp_test[..., np.newaxis], 2, axis=-1)
        varis_inp_train = np.repeat(varis_inp_train[..., np.newaxis], 2, axis=-1)
        varis_inp_test = np.repeat(varis_inp_test[..., np.newaxis], 2, axis=-1)
        values_inp_train = np.repeat(values_inp_train[..., np.newaxis], 2, axis=-1)
        values_inp_test = np.repeat(values_inp_test[..., np.newaxis], 2, axis=-1)

        # Construct training and testing sequences
        trainX, trainY, testX, testY = [], [], [], []
        for i in range(len(train_data) - seq_len - pre_len):
            a1 = train_data[i: i + seq_len + pre_len]
            a2 = varis_inp_train[i: i + seq_len + pre_len]
            a3 = times_inp_train[i: i + seq_len + pre_len]
            a4 = values_inp_train[i: i + seq_len + pre_len]
            a5 = spatial_inp_train[i: i + seq_len + pre_len]
            a = np.column_stack((a1[0:seq_len], a2[0: seq_len], a3[0: seq_len],
                                 a4[0: seq_len], a5[0: seq_len]))
            trainX.append(a)
            trainY.append(a1[seq_len: seq_len + pre_len])

        for i in range(len(test_data) - seq_len - pre_len):
            b1 = test_data[i: i + seq_len + pre_len]
            b2 = varis_inp_test[i: i + seq_len + pre_len]
            b3 = times_inp_test[i: i + seq_len + pre_len]
            b4 = values_inp_test[i: i + seq_len + pre_len]
            b5 = spatial_inp_test[i: i + seq_len + pre_len]
            b = np.column_stack((b1[0:seq_len], b2[0: seq_len], b3[0: seq_len],
                                 b4[0: seq_len], b5[0: seq_len]))
            testX.append(b)
            testY.append(b1[seq_len: seq_len + pre_len])

        trainX1 = np.array(trainX)
        trainY1 = np.array(trainY)
        testX1 = np.array(testX)
        testY1 = np.array(testY)

        return trainX1, trainY1, testX1, testY1