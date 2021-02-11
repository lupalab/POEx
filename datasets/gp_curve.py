import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import os
import pickle
import itertools

data_root = os.path.split(os.path.abspath(__file__))[0]

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

            self.x_size = 1
            self.y_size = 1
            self.split = split
            self.size = 10000 if split =='train' else 5000
            self.num_batches = self.size // batch_size

            inds = tf.range(self.num_batches, dtype=tf.int32)
            dst = tf.data.Dataset.from_tensor_slices(inds)
            dst = dst.map(lambda i: get_batch(i, split, batch_size, set_size), num_parallel_calls=16)
            dst = dst.prefetch(1)
            dst_it = dst.make_initializable_iterator()
            self.xc, self.yc, self.xt, self.yt = dst_it.get_next()
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        xc, yc, xt, yt = self.sess.run([self.xc, self.yc, self.xt, self.yt])

        return {'xc':xc, 'yc':yc, 'xt':xt, 'yt':yt}

class MTDataset(Dataset):
    def next_batch(self):
        batch = super().next_batch()
        xc, yc, xt, yt = batch['xc'], batch['yc'], batch['xt'], batch['yt']
        B,K,Nc,dx = xc.shape
        B,K,Nt,dy = yt.shape
        indexes = np.eye(K)
        ic = np.repeat(np.repeat(np.reshape(indexes, [1,K,1,K]), B, axis=0), Nc, axis=2)
        it = np.repeat(np.repeat(np.reshape(indexes, [1,K,1,K]), B, axis=0), Nt, axis=2)
        xcs, ycs, xts, yts = [], [], [], []
        # f0 = f0
        xc0, yc0, xt0, yt0 = xc[:,0], yc[:,0], xt[:,0], yt[:,0]
        xcs.append(xc0)
        ycs.append(yc0)
        xts.append(xt0)
        yts.append(yt0)
        # f1 = 2-f1
        xc1, yc1, xt1, yt1 = xc[:,1], 2-yc[:,1], xt[:,1], 2-yt[:,1]
        xcs.append(xc1)
        ycs.append(yc1)
        xts.append(xt1)
        yts.append(yt1)
        # f2 = f2(x+1)
        xc2, yc2, xt2, yt2 = xc[:,2]-1, yc[:,2], xt[:,2]-1, yt[:,2]
        xcs.append(xc2)
        ycs.append(yc2)
        xts.append(xt2)
        yts.append(yt2)
        # f3 = f3(x-1)
        xc3, yc3, xt3, yt3 = xc[:,3]+1, yc[:,3], xt[:,3]+1, yt[:,3]
        xcs.append(xc3)
        ycs.append(yc3)
        xts.append(xt3)
        yts.append(yt3)
        # f4 = f4 + 0.5
        xc4, yc4, xt4, yt4 = xc[:,4], yc[:,4]+0.5, xt[:,4], yt[:,4]+0.5
        xcs.append(xc4)
        ycs.append(yc4)
        xts.append(xt4)
        yts.append(yt4)
        # stack
        xc = np.stack(xcs, axis=1)
        xc = np.concatenate([xc, ic], axis=-1)
        yc = np.stack(ycs, axis=1)
        xt = np.stack(xts, axis=1)
        xt = np.concatenate([xt, it], axis=-1)
        yt = np.stack(yts, axis=1)

        batch = {'xc':xc, 'yc':yc, 'xt':xt, 'yt':yt}

        return batch


def get_batch(ind, split, batch_size, set_size):
    if split == 'train':
        num_curve = set_size
        max_num_context = 10
        max_num_target = 50
        num_context = tf.random_uniform(shape=[], minval=1, maxval=max_num_context, dtype=tf.int32)
        num_target = tf.random_uniform(shape=[], minval=10, maxval=max_num_target, dtype=tf.int32)
        num_total_points = num_context + num_target
        xdata = tf.random_uniform([batch_size, num_total_points, 1], -2, 2)
        ydata = gaussian_process(xdata, num_curve) # [B,K,N,1]
        xdata = tf.tile(tf.expand_dims(xdata, axis=1), [1,num_curve,1,1])
    else:
        num_curve = set_size
        max_num_context = 10
        num_context = tf.random_uniform(shape=[], minval=1, maxval=max_num_context, dtype=tf.int32)
        num_total_points = 50
        xdata = tf.range(-2., 2., 4. / num_total_points, dtype=tf.float32)
        xdata = tf.reshape(xdata, [1,num_total_points,1])
        xdata = tf.tile(xdata, [batch_size, 1, 1])
        ydata = gaussian_process(xdata, num_curve) # [B,K,N,1]
        xdata = tf.tile(tf.expand_dims(xdata, axis=1), [1,num_curve,1,1])

    # shuffle
    id0 = tf.tile(tf.reshape(tf.range(batch_size), [batch_size,1,1]), [1,num_curve,num_total_points])
    id1 = tf.tile(tf.reshape(tf.range(num_curve), [1,num_curve,1]), [batch_size,1,num_total_points])
    id2 = tf.stack([tf.tile(tf.reshape(tf.random.shuffle(tf.range(num_total_points)), [1,num_total_points]), [batch_size,1]) for _ in range(num_curve)], axis=1)
    idx = tf.stack([id0, id1, id2], axis=-1)
    xdata = tf.gather_nd(xdata, idx)
    ydata = tf.gather_nd(ydata, idx)

    # sample context
    x_ctx = xdata[:,:,:num_context]
    y_ctx = ydata[:,:,:num_context]

    return x_ctx, y_ctx, xdata, ydata

def gaussian_process(xdata, num_curve):
    batch_size = tf.shape(xdata)[0]
    num_total_points = tf.shape(xdata)[1]

    ### Kx
    l1 = tf.ones([batch_size, 1, 1]) * 0.6 # [B,d,d]
    sigma_f = tf.ones([batch_size, 1]) * 1.0 # [B,d]
    sigma_noise=2e-2
    # Expand and take the difference
    xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
    xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
    diff = xdata1 - xdata2  # [B*K, num_total_points, num_total_points, x_size]

    # [B, y_size, num_total_points, num_total_points, x_size]
    norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

    # [B, y_size, num_total_points, num_total_points]
    norm = tf.reduce_sum(norm, -1)

    # [B, y_size, num_total_points, num_total_points]
    Kx = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)
    Kx = tf.reshape(Kx, [batch_size, 1, 1, num_total_points, 1, num_total_points])

    ### Kt
    corr = 0.9
    Kt = tf.ones([batch_size, num_curve, num_curve], tf.float32) - tf.eye(num_curve)
    Kt = Kt * corr + tf.eye(num_curve)
    Kt = tf.matmul(Kt, Kt)
    Kt = tf.reshape(Kt, [batch_size, 1, num_curve, 1, num_curve, 1])

    ### kernel
    kernel = Kx * Kt
    kernel = tf.reshape(kernel, [batch_size, 1, num_curve*num_total_points, num_curve*num_total_points])
    kernel += (sigma_noise**2) * tf.eye(num_curve*num_total_points)
    cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)

    ### y_values
    y_values = tf.matmul(
        cholesky,
        tf.random_normal([batch_size, 1, num_curve*num_total_points, 1]))
    y_values = tf.reshape(y_values, [batch_size, num_curve, num_total_points, 1])

    return y_values

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = Dataset('train', 32, 5)
    data.initialize()
    batch = data.next_batch()
    xc, yc, xt, yt = batch['xc'], batch['yc'], batch['xt'], batch['yt']
    print(xt.shape)
    print(yt.shape)
    print(xc.shape)
    print(yc.shape)

    B,K,N,d = xt.shape

    fig = plt.figure()
    for i in range(K):
        ax = fig.add_subplot(5, 1, i+1)
        idx = np.argsort(xt[0,i,:,0])
        plt.plot(xt[0,i,idx], yt[0,i,idx], 'b:', linewidth=2)
        plt.plot(xc[0,i], yc[0,i], 'ko', markersize=3)
    plt.savefig('test.png')
    plt.close('all')


    
