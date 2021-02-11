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

def generate_mask(x, mask_type):
    if mask_type == b'expand':
        m = np.zeros_like(x)
        N = np.random.randint(x.shape[0]//8, x.shape[0])
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    elif mask_type == b'arb_expand':
        m = np.zeros_like(x)
        N = np.random.randint(x.shape[0])
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    elif mask_type == b'few_expand':
        m = np.zeros_like(x)
        N = np.random.randint(x.shape[0]//8)
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    elif mask_type == b'det_expand':
        m = np.zeros_like(x)
        N = x.shape[0]//8
        ind = np.random.choice(x.shape[0], N, replace=False)
        m[ind] = 1.
    else:
        raise ValueError()

    return m

def rotate(x):
    angles = np.random.uniform() * 2 * np.pi
    Rz = np.array([[np.cos(angles),-np.sin(angles),0],
                   [np.sin(angles),np.cos(angles),0],
                   [0,0,1]])

    rotated_data = np.dot(x, Rz).astype(np.float32)

    return rotated_data

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
    # rotate
    x = rotate(x)
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

def get_dst(split, set_size, category, mask_type):
    assert category in label_dict
    if split == 'train':
        inds = np.where(train_labels == label_dict[category])[0]
        size = len(inds)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, set_size, mask_type], 
            [tf.float32, tf.float32])),
            num_parallel_calls=16)
    else:
        inds = np.where(test_labels == label_dict[category])[0]
        size = len(inds)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, set_size, mask_type], 
            [tf.float32, tf.float32])),
            num_parallel_calls=16)

    return dst, size

class Dataset(object):
    def __init__(self, split, batch_size, set_size, category, mask_type):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            dst, size = get_dst(split, set_size, category, mask_type)
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

class mDataset(Dataset):
    def next_batch(self):
        x, b = self.sess.run([self.x, self.b])
        m = b.copy()
        for i in range(b.shape[0]):
            ind = np.random.choice(np.where(b[i,:,0]==0)[0])
            m[i,ind] = 1.0

        return {'x':x, 'b':b, 'm':m}