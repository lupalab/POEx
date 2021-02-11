import os
import pickle
import numpy as np
import tensorflow as tf
from collections import Counter

# load data
data_path = '/data/molecule/qm9.pkl'
with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)
num_atoms = data_dict['num_atoms']
atom_types = data_dict['atom_types']
point_cloud = data_dict['point_cloud']
dict_typs = {0:'H', 1:'C', 2:'N', 3:'O', 4:'F'}

# split
rng = np.random.RandomState(42)
ids = rng.permutation(len(data_dict['num_atoms']))
train_ids = ids[:int(len(ids)*0.8)]
valid_ids = ids[int(len(ids)*0.8):int(len(ids)*0.9)]
test_ids = ids[int(len(ids)*0.9):]

_SIZE = {
    'train': len(train_ids),
    'valid': len(valid_ids),
    'test': len(test_ids)
}

# stats
train_counter = Counter(num_atoms[train_ids])
valid_counter = Counter(num_atoms[valid_ids])
test_counter = Counter(num_atoms[test_ids])

def gen_mask(x, mask_type):
    '''
    x: [N,d]
    m: [N,d]
    '''
    if mask_type == b'impute':
        m = np.zeros_like(x)
        O = np.random.randint(1, x.shape[0])
        ind = np.random.choice(x.shape[0], O, replace=False)
        m[ind] = 1.
    elif mask_type == b'joint':
        m = np.zeros_like(x)

    return m

def center(x):
    mu = np.mean(x, axis=0)

    return x - mu

def rotate(x):
    angles = np.random.uniform(size=3) * 2 * np.pi
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))

    rotated_data = np.dot(x, R)

    return rotated_data

def get_batch(split, batch_size, mask_type):
    if split == b'train':
        s = list(train_counter.keys())
        p = np.array(list(train_counter.values()))
        p = p / np.sum(p)
        set_size = np.random.choice(s, p=p)
        set_ids = np.where(num_atoms[train_ids]==set_size)[0]
        batch_ids = np.random.choice(set_ids, size=batch_size, replace=len(set_ids)<batch_size)
        batch_ids = train_ids[batch_ids]
        pcs = [point_cloud[i] for i in batch_ids]
        pcs = [rotate(center(pc)) for pc in pcs]
        typ = [atom_types[i] for i in batch_ids]
        msk = [gen_mask(pc,mask_type) for pc in pcs]
    elif split == b'valid':
        s = list(valid_counter.keys())
        p = np.array(list(valid_counter.values()))
        p = p / np.sum(p)
        set_size = np.random.choice(s, p=p)
        set_ids = np.where(num_atoms[valid_ids]==set_size)[0]
        batch_ids = np.random.choice(set_ids, size=batch_size, replace=len(set_ids)<batch_size)
        batch_ids = valid_ids[batch_ids]
        pcs = [point_cloud[i] for i in batch_ids]
        pcs = [center(pc) for pc in pcs]
        typ = [atom_types[i] for i in batch_ids]
        msk = [gen_mask(pc,mask_type) for pc in pcs]
    elif split == b'test':
        s = list(test_counter.keys())
        p = np.array(list(test_counter.values()), dtype=np.float32)
        p = p / np.sum(p)
        set_size = np.random.choice(s, p=p)
        set_ids = np.where(num_atoms[test_ids]==set_size)[0]
        batch_ids = np.random.choice(set_ids, size=batch_size, replace=len(set_ids)<batch_size)
        batch_ids = test_ids[batch_ids]
        pcs = [point_cloud[i] for i in batch_ids]
        pcs = [center(pc) for pc in pcs]
        typ = [atom_types[i] for i in batch_ids]
        msk = [gen_mask(pc,mask_type) for pc in pcs]
    else:
        raise ValueError()

    typ = np.eye(5)[np.stack(typ)].astype(np.float32) #[B,N,5]
    pcs = np.stack(pcs).astype(np.float32) #[B,N,3]
    msk = np.stack(msk).astype(np.float32) #[B,N,3]

    return typ, pcs, msk

class Dataset(object):
    def __init__(self, split, batch_size, mask_type):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            self.size = _SIZE[split]
            self.num_batches = self.size // batch_size
            dst = tf.data.Dataset.from_tensor_slices(tf.range(self.num_batches))
            dst = dst.map(lambda i: tuple(
                tf.py_func(get_batch, [split, batch_size, mask_type], [tf.float32, tf.float32, tf.float32])),
                num_parallel_calls=16)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            self.t, self.x, self.b  = dst_it.get_next()
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        t, x, b = self.sess.run([self.t, self.x, self.b])
        m = np.ones_like(b)
        return {'t':t, 'x':x, 'b':b, 'm':m}