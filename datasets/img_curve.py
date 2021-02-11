import os
import torch
import pickle
import numpy as np
import itertools
import tensorflow as tf

H, W, C = image_shape = [28, 28, 1]
N = H * W
h = np.arange(H, dtype=np.float32) / (H - 1)
w = np.arange(W, dtype=np.float32) / (W - 1)
indexes = np.array(list(itertools.product(h, w))) # [H*W,2]

data_path = '/data/mnist/'
train_x, train_y = torch.load(data_path + 'training.pt')
train_x, train_y = train_x.numpy(), train_y.numpy()
test_x, test_y = torch.load(data_path + 'test.pt')
test_x, test_y = test_x.numpy(), test_y.numpy()

class Dataset(object):
    def __init__(self, split, batch_size, set_size):
        super().__init__()

        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)

            self.x_size = 2
            self.y_size = 1
            self.split = split
            self.size = 10000 if split =='train' else 1000
            self.num_batches = self.size // batch_size

            inds = tf.range(self.num_batches, dtype=tf.int32)
            dst = tf.data.Dataset.from_tensor_slices(inds)
            dst = dst.map(lambda i: tuple(
                tf.py_func(get_batch, [i, split, batch_size, set_size],
                [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32])),
                num_parallel_calls=16)
            dst = dst.prefetch(1)
            dst_it = dst.make_initializable_iterator()
            self.idx, self.xc, self.yc, self.xt, self.yt = dst_it.get_next()
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        idx, xc, yc, xt, yt = self.sess.run([self.idx, self.xc, self.yc, self.xt, self.yt])

        return {'idx':idx, 'xc':xc, 'yc':yc, 'xt':xt, 'yt':yt}

def get_batch(ind, split, batch_size, set_size):
    data = train_x if split == b'train' else test_x
    label = train_y if split == b'train' else test_y
    image = []
    for _ in range(batch_size):
        lab = np.random.choice(10)
        inds = np.where(label==lab)[0]
        ind = np.random.choice(inds, size=set_size, replace=False)
        image.append(data[ind])
    image = np.stack(image).reshape([batch_size, set_size, N, 1]) #[B,K,H*W,1]
    image = image.astype(np.float32) / 255. - 0.5
    index = np.reshape(indexes, [1,1,N,2])
    index = np.repeat(np.repeat(index, set_size, axis=1), batch_size, axis=0) #[B,K,H*W,2]

    num_context = np.random.randint(50, 200)
    idx = np.random.choice(N, num_context, replace=False)

    # sample context
    x_ctx = index[:,:,idx]
    y_ctx = image[:,:,idx]

    return idx, x_ctx, y_ctx, index, image
    
