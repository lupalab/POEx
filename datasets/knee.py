import os
import numpy as np
import scipy.sparse as sparse
import pickle
import h5py
import tensorflow as tf
from PIL import Image

image_shape = [64, 64, 2]
data_path = '/data/fastMRI/knee/'

def load_data(files):
    data = []
    for f in files:
        tmp = []
        with h5py.File(f, 'r') as hf:
            x = hf['reconstruction_esc'][:]
        num_slices = x.shape[0]
        for xs in x[num_slices//4:3*num_slices//4]:
            xs = xs[96:-96, 96:-96]
            tmp.append(sparse.coo_matrix(xs))
        data.append(tmp)
    return data

if os.path.isfile(data_path+'volume128.pkl'):
    with open(data_path+'volume128.pkl', 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f)
else:
    train_folder = data_path+'singlecoil_train/'
    train_files = os.listdir(train_folder)[:-50]
    train_files = [train_folder+f for f in train_files]
    valid_files = os.listdir(train_folder)[-50:]
    valid_files = [train_folder+f for f in valid_files]
    test_folder = data_path+'singlecoil_val/'
    test_files = os.listdir(test_folder)
    test_files = [test_folder+f for f in test_files]
    
    train_data = load_data(train_files)
    valid_data = load_data(valid_files)
    test_data = load_data(test_files)
    with open(data_path+'volume128.pkl', 'wb') as f:
        pickle.dump((train_data, valid_data, test_data), f)

def position_embedding(position, embed_dim):
    div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
    e0 = np.sin(position * div_term)
    e1 = np.cos(position * div_term)
    e = np.stack([e0, e1], axis=-1).reshape([-1,embed_dim])

    return e.astype(np.float32)

def cartesian_mask(image, max_acquisition, center_acquisition):
    mask = np.zeros_like(image)
    H = image.shape[0]
    N = max_acquisition-center_acquisition
    pad = (H - center_acquisition + 1) // 2
    center_idx = range(pad, pad+center_acquisition)
    choices = list(set(range(H)) - set(center_idx))
    idx = np.random.choice(choices, N, replace=False)
    mask[center_idx] = 1
    mask[idx] = 1
    return mask

def generate_mask(image):
    mask = np.stack([cartesian_mask(im, 8, 2) for im in image], axis=0)
    return mask

def to_kspace(x):
    x = np.fft.fftshift(np.fft.fft2(x),axes=(-2,-1))
    x = np.stack([np.real(x), np.imag(x)], axis=-1)
    x = x.astype(np.float32)
    return x

def resize(x):
    im = Image.fromarray(x).resize((64,64))
    x = np.array(im)
    return x

def normalize(x):
    x = x.astype(np.float32)
    xmin = np.min(x)
    xmax = np.max(x)
    return (x - xmin) / (xmax - xmin)

def _parse_train(i, set_size, t_dim):
    slices = train_data[i]
    assert len(slices) > set_size
    s = np.random.randint(0, len(slices)-set_size)
    x = [resize(slices[k].todense()) for k in range(s, s+set_size)]
    x = np.stack(x, axis=0) #[N,H,W]
    x = normalize(x)
    x = to_kspace(x) # [N,H,W,2]
    t = np.expand_dims(np.arange(set_size), axis=-1) # [N,1]
    t = position_embedding(t, t_dim)
    b = generate_mask(x)
    m = np.ones_like(b)

    return t, x, b, m

def _parse_valid(i, set_size, t_dim):
    slices = valid_data[i]
    assert len(slices) > set_size
    s = np.random.randint(0, len(slices)-set_size)
    x = [resize(slices[k].todense()) for k in range(s, s+set_size)]
    x = np.stack(x, axis=0) #[N,H,W]
    x = normalize(x)
    x = to_kspace(x) # [N,H,W,2]
    t = np.expand_dims(np.arange(set_size), axis=-1) # [N,1]
    t = position_embedding(t, t_dim)
    b = generate_mask(x)
    m = np.ones_like(b)

    return t, x, b, m

def _parse_test(i, set_size, t_dim):
    slices = test_data[i]
    assert len(slices) > set_size
    s = np.random.randint(0, len(slices)-set_size)
    x = [resize(slices[k].todense()) for k in range(s, s+set_size)]
    x = np.stack(x, axis=0) #[N,H,W]
    x = normalize(x)
    x = to_kspace(x) # [N,H,W,2]
    t = np.expand_dims(np.arange(set_size), axis=-1) # [N,1]
    t = position_embedding(t, t_dim)
    b = generate_mask(x)
    m = np.ones_like(b)

    return t, x, b, m

def get_dst(split, set_size, t_dim):
    if split == 'train':
        size = len(train_data)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, set_size, t_dim],
                       [tf.float32, tf.float32, tf.float32, tf.float32])),
                      num_parallel_calls=16)
    elif split == 'valid':
        size = len(valid_data)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i, set_size, t_dim],
                       [tf.float32, tf.float32, tf.float32, tf.float32])),
                      num_parallel_calls=16)
    else:
        size = len(test_data)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, set_size, t_dim],
                       [tf.float32, tf.float32, tf.float32, tf.float32])),
                      num_parallel_calls=16)
    
    return dst, size

class Dataset(object):
    def __init__(self, split, batch_size, set_size, t_dim):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            dst, size = get_dst(split, set_size, t_dim)
            self.size = size
            self.num_batches = self.size // batch_size
            dst = dst.batch(batch_size, drop_remainder=True)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            t, x, b, m  = dst_it.get_next()
            self.t = tf.reshape(t, [batch_size, set_size, t_dim])
            self.x = tf.reshape(x, [batch_size, set_size] + image_shape)
            self.b = tf.reshape(b, [batch_size, set_size] + image_shape)
            self.m = tf.reshape(m, [batch_size, set_size] + image_shape)
            self.image_shape = image_shape
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        t, x, b, m = self.sess.run([self.t, self.x, self.b, self.m])
        return {'t':t, 'x':x, 'b':b, 'm':m}