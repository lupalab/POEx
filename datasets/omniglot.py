import os
import pickle
import numpy as np
import tensorflow as tf

from .mask_generators import *

# load data
image_shape = [32,32,1]
data_path = '/data/omniglot/'
with open(data_path + 'train_vinyals_aug90.pkl', 'rb') as f:
    train_dict = pickle.load(f, encoding='bytes')
    num_train_labels = len(train_dict[b'label_str'])
with open(data_path + 'val_vinyals_aug90.pkl', 'rb') as f:
    valid_dict = pickle.load(f, encoding='bytes')
    num_valid_labels = len(valid_dict[b'label_str'])
with open(data_path + 'test_vinyals_aug90.pkl', 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')
    num_test_labels = len(test_dict[b'label_str'])
# merge train and valid
train_dict[b'images'] = np.concatenate([train_dict[b'images'], valid_dict[b'images']])
train_dict[b'labels'] = np.concatenate([train_dict[b'labels'], valid_dict[b'labels']+num_train_labels])
train_dict[b'label_str'] = train_dict[b'label_str'] + valid_dict[b'label_str']
num_train_labels = num_train_labels + num_valid_labels

# mask generation
mask_fn = ReverseCutoutGenerator(10)
def generate_mask(image, mask_type):
    N,H,W,C = image.shape
    mask = []
    if mask_type == b'impute':
        mask = [mask_fn(im) for im in image]
        mask = np.stack(mask, axis=0)
    elif mask_type == b'expand':
        O = np.random.randint(1,N)
        mask = [np.ones([H,W,C],dtype=np.uint8)] * O
        mask+= [np.zeros([H,W,C],dtype=np.uint8)] * (N-O)
        mask = np.stack(mask, axis=0)
        np.random.shuffle(mask)
    else:
        raise ValueError()

    return mask

def _parse_train(i, set_size, mask_type):
    inds = np.where(train_dict[b'labels']==i)[0][:10]
    ind = np.random.choice(inds, size=set_size, replace=False)
    image = train_dict[b'images'][ind]
    image = np.pad(image, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    mask = generate_mask(image, mask_type)

    return image, mask

def _parse_valid(i, set_size, mask_type):
    inds = np.where(train_dict[b'labels']==i)[0][10:]
    ind = np.random.choice(inds, size=set_size, replace=False)
    image = train_dict[b'images'][ind]
    image = np.pad(image, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    mask = generate_mask(image, mask_type)

    return image, mask

def _parse_test(i, set_size, mask_type):
    inds = np.where(test_dict[b'labels']==i)[0]
    ind = np.random.choice(inds, size=set_size, replace=False)
    image = test_dict[b'images'][ind]
    image = np.pad(image, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    mask = generate_mask(image, mask_type)

    return image, mask

def get_dst(split, set_size, mask_type):
    if split == 'train':
        size = num_train_labels
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, set_size, mask_type], [tf.uint8, tf.uint8])),
            num_parallel_calls=16)
    elif split == 'valid':
        size = num_train_labels
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i, set_size, mask_type], [tf.uint8, tf.uint8])),
            num_parallel_calls=16)
    else:
        size = num_test_labels
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, set_size, mask_type], [tf.uint8, tf.uint8])),
            num_parallel_calls=16)

    return dst, size

class Dataset(object):
    def __init__(self, split, batch_size, set_size, mask_type):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            dst, size = get_dst(split, set_size, mask_type)
            self.size = size
            self.num_batches = self.size // batch_size
            dst = dst.batch(batch_size, drop_remainder=True)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            x, b  = dst_it.get_next()
            self.x = tf.reshape(x, [batch_size, set_size] + image_shape)
            self.b = tf.reshape(b, [batch_size, set_size] + image_shape)
            self.image_shape = image_shape
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        x, b = self.sess.run([self.x, self.b])
        m = np.ones_like(b)
        return {'x':x, 'b':b, 'm':m}
