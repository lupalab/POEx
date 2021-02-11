import h5py
import numpy as np
import tensorflow as tf

with h5py.File('/data/pointcloud/ModelNet40_cloud.h5', 'r') as f:
    train_cloud = np.array(f['tr_cloud'])
    train_labels = np.array(f['tr_labels'])
    test_cloud = np.array(f['test_cloud'])
    test_labels = np.array(f['test_labels'])

label_dict = {'airplane': 0,
              'bathtub': 1,
              'bed': 2,
              'bench': 3,
              'bookshelf': 4,
              'bottle': 5,
              'bowl': 6,
              'car': 7,
              'chair': 8,
              'cone': 9,
              'cup': 10,
              'curtain': 11,
              'desk': 12,
              'door': 13,
              'dresser': 14,
              'flower_pot': 15,
              'glass_box': 16,
              'guitar': 17,
              'keyboard': 18,
              'lamp': 19,
              'laptop': 20,
              'mantel': 21,
              'monitor': 22,
              'night_stand': 23,
              'person': 24,
              'piano': 25,
              'plant': 26,
              'radio': 27,
              'range_hood': 28,
              'sink': 29,
              'sofa': 30,
              'stairs': 31,
              'stool': 32,
              'table': 33,
              'tent': 34,
              'toilet': 35,
              'tv_stand': 36,
              'vase': 37,
              'wardrobe': 38,
              'xbox': 39,
              }

train_inds = []
valid_inds = []
test_inds = []
for i in range(30):
    inds = np.where(train_labels == i)[0]
    N = int(len(inds) * 0.8)
    train_inds.append(inds[:N])
    valid_inds.append(inds[N:])
for i in range(30,40):
    inds = np.where(test_labels == i)[0]
    test_inds.append(inds)
train_inds = np.concatenate(train_inds)
valid_inds = np.concatenate(valid_inds)
test_inds = np.concatenate(test_inds)
print(f'train:{len(train_inds)} valid:{len(valid_inds)} test:{len(test_inds)}')

def generate_mask(x, mask_type):
    if mask_type == b'expand':
        m = np.zeros_like(x)
        N = np.random.randint(x.shape[0]//8, x.shape[0])
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    elif mask_type == b'few_expand':
        m = np.zeros_like(x)
        N = np.random.randint(x.shape[0]//8)
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    elif mask_type == b'arb_expand':
        m = np.zeros_like(x)
        N = np.random.randint(x.shape[0])
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    elif mask_type == b'det_expand':
        m = np.zeros_like(x)
        ind = np.random.choice(x.shape[0], 256, replace=False)
        m[ind] = 1.
    elif mask_type == b'complete':
        m = np.zeros_like(x)
        while np.sum(m[:,0]) < x.shape[0] // 8:
            p = np.random.uniform(-0.5, 0.5, size=4)
            xa = np.concatenate([x, np.ones([x.shape[0],1])], axis=1)
            m = (np.dot(xa, p) > 0).astype(np.float32)
            m = np.repeat(np.expand_dims(m, axis=1), 3, axis=1)
    else:
        raise ValueError()

    return m

def _parse_train(i, set_size, mask_type):
    x = train_cloud[i].astype(np.float32)
    # subsample
    ind = np.random.choice(x.shape[0], set_size, replace=False)
    x = x[ind]
    # preprocess
    x += np.random.randn(*x.shape) * 0.001
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x - x_min) / (x_max - x_min)
    x -= np.mean(x, axis=0)
    # generate mask
    m = generate_mask(x, mask_type)

    return x, m

def _parse_valid(i, set_size, mask_type):
    x = train_cloud[i].astype(np.float32)
    # subsample
    ind = np.random.choice(x.shape[0], set_size, replace=False)
    x = x[ind]
    # preprocess
    x += np.random.randn(*x.shape) * 0.001
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x - x_min) / (x_max - x_min)
    x -= np.mean(x, axis=0)
    # generate mask
    m = generate_mask(x, mask_type)

    return x, m

def _parse_test(i, set_size, mask_type):
    x = test_cloud[i].astype(np.float32)
    # subsample
    ind = np.random.choice(x.shape[0], set_size, replace=False)
    x = x[ind]
    # preprocess
    x += np.random.randn(*x.shape) * 0.001
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x - x_min) / (x_max - x_min)
    x -= np.mean(x, axis=0)
    # generate mask
    m = generate_mask(x, mask_type)

    return x, m

def get_dst(split, set_size, mask_type):
    if split == 'train':
        inds = train_inds
        size = len(inds)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, set_size, mask_type], [tf.float32, tf.float32])),
            num_parallel_calls=16)
    elif split == 'valid':
        inds = valid_inds
        size = len(inds)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i, set_size, mask_type], [tf.float32, tf.float32])),
            num_parallel_calls=16)
    else:
        inds = test_inds
        size = len(inds)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, set_size, mask_type], [tf.float32, tf.float32])),
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
            self.x = tf.reshape(x, [batch_size, set_size, 3])
            self.b = tf.reshape(b, [batch_size, set_size, 3])
            self.dimension = 3
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        x, b = self.sess.run([self.x, self.b])
        m = np.ones_like(b)
        return {'x':x, 'b':b, 'm':m}