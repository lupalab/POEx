import os
import pickle
import numpy as np
import tensorflow as tf

from .mask_generators import *

image_shape = [32, 32, 1]
data_path = '/data/mnist/'
train_x, train_y = torch.load(data_path + 'training.pt')
train_x, train_y = train_x.numpy(), train_y.numpy()
train_x = np.pad(train_x, ((0, 0), (2, 2), (2, 2)), mode='constant')
train_x = train_x[:, :, :, np.newaxis]
test_x, test_y = torch.load(data_path + 'test.pt')
test_x, test_y = test_x.numpy(), test_y.numpy()
test_x = np.pad(test_x, ((0, 0), (2, 2), (2, 2)), mode='constant')
test_x = test_x[:, :, :, np.newaxis]

data = np.concatenate([train_x, test_x], axis=0)
label = np.concatenate([train_y, test_y], axis=0)

# mask generator
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
    lab = np.random.randint(10) // 2 * 2
    inds = np.where(label==lab)[0]
    inds = inds[:int(0.9*len(inds))]
    ind = np.random.choice(inds, size=set_size, replace=False)
    image = data[ind]
    mask = generate_mask(image, mask_type)

    return image, mask

def _parse_valid(i, set_size, mask_type):
    lab = np.random.randint(10) // 2 * 2
    inds = np.where(label==lab)[0]
    inds = inds[int(0.9*len(inds)):]
    ind = np.random.choice(inds, size=set_size, replace=False)
    image = data[ind]
    mask = generate_mask(image, mask_type)

    return image, mask

def _parse_test(i, set_size, mask_type):
    lab = np.random.randint(10) // 2 * 2 + 1
    inds = np.where(label==lab)[0]
    ind = np.random.choice(inds, size=set_size, replace=False)
    image = data[ind]
    mask = generate_mask(image, mask_type)

    return image, mask

def get_dst(split, set_size, mask_type):
    if split == 'train':
        size = 10000
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, set_size, mask_type], [tf.uint8, tf.uint8])),
            num_parallel_calls=16)
    elif split == 'valid':
        size = 5000
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i, set_size, mask_type], [tf.uint8, tf.uint8])),
            num_parallel_calls=16)
    else:
        size = 5000
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