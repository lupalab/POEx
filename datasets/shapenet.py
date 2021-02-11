import os
import pickle
import numpy as np
import tensorflow as tf

# load data
image_shape = [32,32,1]
data_path = '/data/shapenet/data.pkl'
with open(data_path, 'rb') as f:
    train, valid, test = pickle.load(f)

def generate_mask(image):
    N,H,W,C = image.shape
    mask = []
    O = np.random.randint(1,N)
    mask += [np.ones([H,W,C],dtype=np.uint8)] * O
    mask += [np.zeros([H,W,C], dtype=np.uint8)] * (N-O)
    mask = np.stack(mask, axis=0)
    np.random.shuffle(mask)

    return mask

def get_angle(index):
    angle = index * 10.
    angle_radians = np.deg2rad(angle)
    t = np.stack([np.sin(angle_radians), np.cos(angle_radians)], axis=-1)

    return t.astype(np.float32)
    
def _parse_train(i, set_size):
    ind = np.random.choice(36, size=set_size, replace=False)
    image = train[i,ind].reshape([set_size]+image_shape)
    mask = generate_mask(image)
    angle = get_angle(ind)

    return angle, image, mask

def _parse_valid(i, set_size):
    ind = np.random.choice(36, size=set_size, replace=False)
    image = valid[i,ind].reshape([set_size]+image_shape)
    mask = generate_mask(image)
    angle = get_angle(ind)

    return angle, image, mask

def _parse_test(i, set_size):
    ind = np.random.choice(36, size=set_size, replace=False)
    image = test[i,ind].reshape([set_size]+image_shape)
    mask = generate_mask(image)
    angle = get_angle(ind)

    return angle, image, mask

def get_dst(split, set_size):
    if split == 'train':
        size = len(train)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, set_size], [tf.float32, tf.uint8, tf.uint8])),
            num_parallel_calls=16)
    elif split == 'valid':
        size = len(valid)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i, set_size], [tf.float32, tf.uint8, tf.uint8])),
            num_parallel_calls=16)
    else:
        size = len(test)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, set_size], [tf.float32, tf.uint8, tf.uint8])),
            num_parallel_calls=16)

    return dst, size

class Dataset(object):
    def __init__(self, split, batch_size, set_size):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            dst, size = get_dst(split, set_size)
            self.size = size
            self.num_batches = self.size // batch_size
            dst = dst.batch(batch_size, drop_remainder=True)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            t, x, b  = dst_it.get_next()
            self.t = tf.reshape(t, [batch_size, set_size, 2])
            self.x = tf.reshape(x, [batch_size, set_size] + image_shape)
            self.b = tf.reshape(b, [batch_size, set_size] + image_shape)
            self.image_shape = image_shape
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        t, x, b = self.sess.run([self.t, self.x, self.b])
        m = np.ones_like(b)
        return {'t':t, 'x':x, 'b':b, 'm':m}