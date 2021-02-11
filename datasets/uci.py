import os
import pickle
import numpy as np
import tensorflow as tf

data_path = '/data/maf/'

_NAME = {
    'gas': 'maf_gas.p',
    'bsda': 'maf_bsds.p',
    'power': 'maf_power.p',
    'hepmass': 'maf_hepmass.p',
    'miniboone': 'maf_miniboone.p'
}

def load_data(dname, split):
    with open(f'{data_path}/{_NAME[dname]}', 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    mean = data_dict['train'].mean(axis=0)
    std = data_dict['train'].std(axis=0)
    data = ((data_dict[split] - mean) / std).astype(np.float32)
    return data

def get_mask(x):
    m = np.random.choice([0,1], size=x.shape)
    return m.astype(np.float32)

class Dataset(object):
    def __init__(self, dname, split, batch_size):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            data = load_data(dname, split)
            self.dimension = data.shape[1]
            self.size = data.shape[0]
            self.num_batches = self.size // batch_size
            # self.num_batches = 100
            dst = tf.data.Dataset.from_tensor_slices(data)
            if split == 'train':
                dst = dst.shuffle(self.size)
            dst = dst.batch(batch_size, drop_remainder=True)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            self.x = dst_it.get_next()
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        x = self.sess.run(self.x)
        x = np.expand_dims(x, axis=-1) #[B,N,1]
        b = get_mask(x)
        m = np.ones_like(b)
        t = np.repeat(np.expand_dims(np.eye(self.dimension), axis=0), x.shape[0], axis=0) #[B,N,d]

        return {'t':t, 'x':x, 'b':b, 'm':m}
