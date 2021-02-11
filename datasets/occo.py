import numpy as np
import tensorflow as tf
from tensorpack import dataflow

data_path = '/data/occo/'

def resample_pcd(pcd, n):
    """drop or duplicate points so that input of each object has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

class BatchData(dataflow.ProxyDataFlow):
    def __init__(self, ds, batch_size, input_size, gt_size, remainder=False, use_list=False):
        super(BatchData, self).__init__(ds)
        self.batch_size = batch_size
        self.input_size = input_size
        self.gt_size = gt_size
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        """get the number of batches"""
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)  # int(False) == 0

    def __iter__(self):
        """generating data in batches"""
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch(holder, self.use_list)
                del holder[:]  # reset holder as empty list => holder = []
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch(holder, self.use_list)

    def _aggregate_batch(self, data_holder, use_list=False):
        ids = np.stack([x[0] for x in data_holder])
        inputs = np.stack([resample_pcd(x[1], self.input_size) for x in data_holder]).astype(np.float32)
        mask_inputs = np.ones_like(inputs)
        gts = np.stack([resample_pcd(x[2], self.gt_size) for x in data_holder]).astype(np.float32)
        mask_gts = np.zeros_like(gts)
        x = np.concatenate([inputs, gts], axis=1)
        m = np.concatenate([mask_inputs, mask_gts], axis=1)
        # normalization
        x += np.random.randn(*x.shape) * 0.01
        x_max = np.max(x, axis=1, keepdims=True)
        x_min = np.min(x, axis=1, keepdims=True)
        x = (x - x_min) / (x_max - x_min)
        x -= np.mean(x, axis=1, keepdims=True)
        return x, m

class Dataset(object):
    def __init__(self, split, batch_size, set_size):
        if split == 'train':
            lmdb_path = f'{data_path}/ModelNet40_train_1024_middle.lmdb'
        else:
            lmdb_path = f'{data_path}/ModelNet40_test_1024_middle.lmdb'
        df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
        self.size = df.size()
        self.num_batches = self.size // batch_size
        if split == 'train':
            df = dataflow.LocallyShuffleData(df, buffer_size=2000)  # buffer_size
            df = dataflow.PrefetchData(df, num_prefetch=500, num_proc=1)
        df = BatchData(df, batch_size, set_size // 8, set_size - set_size // 8)
        if split == 'train':
            df = dataflow.PrefetchDataZMQ(df, num_proc=8)
        df = dataflow.RepeatedData(df, -1)
        df.reset_state()
        self.generator = df.get_data()

    def initialize(self):
        pass

    def next_batch(self):
        x, b = next(self.generator)
        m = np.ones_like(b)
        return {'x':x, 'b':b, 'm':m}

