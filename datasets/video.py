import os
import gzip
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image

from .mask_generators import *

# load data
image_shape = [64,64,3]
data_path = '/data/occlusion/occlusion_data.pgz'
with gzip.open(data_path, 'rb') as f:
    train_list, test_list = pickle.load(f)

n_train = int(len(train_list) * 0.9)
np.random.shuffle(train_list)
train = train_list[:n_train]
valid = train_list[n_train:]
test = test_list

# mask generation
# mask_fn = VideoMaskGenerator()
mask_fn = RectangleGenerator()
def generate_mask(image, miss):
    mask = np.stack([mask_fn(im) for im in image], axis=0)
    mask = mask * miss

    return mask

def resize(x, is_mask=True):
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    x = Image.fromarray(x).resize((64,64), resample=resample)
    x = np.array(x)

    return x

def position_embedding(position, embed_dim):
    div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
    e0 = np.sin(position * div_term)
    e1 = np.cos(position * div_term)
    e = np.stack([e0, e1], axis=-1).reshape([-1,embed_dim])

    return e.astype(np.float32)

def get_time(index, total_frames, embed_dim):
    index = np.array(index, dtype=np.float32)
    time = (index * 1.0 / total_frames).astype(np.float32)
    time = np.expand_dims(time, axis=-1)

    if embed_dim <= 1:
        return time
    return position_embedding(time, embed_dim)

def _parse_train(i, set_size, t_dim):
    frames = train[i]
    replace = len(frames) < set_size
    inds = np.random.choice(len(frames), set_size, replace=replace)
    image = np.stack([resize(frames[ind][0], False) for ind in inds], axis=0)
    miss = np.stack([1-resize(frames[ind][1], True) for ind in inds], axis=0)
    miss = np.repeat(np.expand_dims(miss, axis=-1), 3, axis=-1)
    mask = generate_mask(image, miss)
    time = get_time(inds, len(frames), t_dim)

    return time, image, mask, miss

def _parse_valid(i, set_size, t_dim):
    frames = valid[i]
    replace = len(frames) < set_size
    inds = np.random.choice(len(frames), set_size, replace=replace)
    image = np.stack([resize(frames[ind][0], False) for ind in inds], axis=0)
    miss = np.stack([1-resize(frames[ind][1], True) for ind in inds], axis=0)
    miss = np.repeat(np.expand_dims(miss, axis=-1), 3, axis=-1)
    mask = generate_mask(image, miss)
    time = get_time(inds, len(frames), t_dim)

    return time, image, mask, miss

def _parse_test(i, set_size, t_dim):
    frames = test[i]
    replace = len(frames) < set_size
    inds = np.random.choice(len(frames), set_size, replace=replace)
    image = np.stack([resize(frames[ind][0], False) for ind in inds], axis=0)
    miss = np.stack([1-resize(frames[ind][1], True) for ind in inds], axis=0)
    miss = np.repeat(np.expand_dims(miss, axis=-1), 3, axis=-1)
    mask = generate_mask(image, miss)
    time = get_time(inds, len(frames), t_dim)

    return time, image, mask, miss

def get_dst(split, set_size, t_dim):
    if split == 'train':
        size = len(train)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, set_size, t_dim], 
            [tf.float32, tf.uint8, tf.uint8, tf.uint8])),
            num_parallel_calls=16)
    elif split == 'valid':
        size = len(valid)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i, set_size, t_dim], 
            [tf.float32, tf.uint8, tf.uint8, tf.uint8])),
            num_parallel_calls=16)
    else:
        size = len(test)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, set_size, t_dim], 
            [tf.float32, tf.uint8, tf.uint8, tf.uint8])),
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