import os
from glob import glob
import numpy as np
import open3d
import tensorflow as tf

data_path = '/data/Colon/goodmesh/'
file_list = glob(f'{data_path}/*/*.obj')
# load meshes
meshes = []
for fname in file_list:
    mesh = open3d.io.read_triangle_mesh(fname)
    meshes.append(mesh)
train_meshes = meshes[:-5]
test_meshes = meshes[-5:]

def position_embedding(position, embed_dim):
    div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
    e0 = np.sin(position * div_term)
    e1 = np.cos(position * div_term)
    e = np.stack([e0, e1], axis=-1).reshape([-1,embed_dim])

    return e.astype(np.float32)

def get_coord(x, num_blocks):
    assert np.max(x) <= 1.0
    coord = np.floor(x * num_blocks) / num_blocks

    return coord

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

    return rotated_data.astype(np.float32)

def sample_data(mesh, num_points):
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    x = np.asarray(pcd.points, dtype=np.float32)
    # preprocess
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x - x_min) / (x_max - x_min)
    x -= np.mean(x, axis=0)
    # rotation
    x = rotate(x)

    return x

def generate_mask(data, mask_type):
    if mask_type == b'complete':
        m = np.zeros_like(data)
        while np.sum(m[:,0]) < data.shape[0] // 8:
            p = np.random.uniform(-0.5, 0.5, size=4)
            x = np.concatenate([data, np.ones([data.shape[0],1])], axis=1)
            m = (np.dot(x, p) > 0).astype(np.float32)
            m = np.repeat(np.expand_dims(m, axis=1), 3, axis=1)
    elif mask_type == b'block':
        radius = np.random.uniform(0.5)
        coord = get_coord(data, 10)
        anchor = coord[np.random.choice(len(data))]
        dist = np.sqrt(np.sum(np.square(coord - anchor), axis=1))
        m = (dist > radius).astype(np.float32)
        m = np.repeat(np.expand_dims(m, axis=1), 3, axis=1)
    else:
        raise ValueError()

    return m

def _parse_train(i, set_size, mask_type):
    i = i % len(train_meshes)
    mesh = train_meshes[i]
    x = sample_data(mesh, set_size)
    m = generate_mask(x, mask_type)
    t = get_coord(x, 10)

    return t, x, m

def _parse_test(i, set_size, mask_type):
    i = i % len(test_meshes)
    mesh = test_meshes[i]
    x = sample_data(mesh, set_size)
    m = generate_mask(x, mask_type)
    t = get_coord(x, 10)

    return t, x, m

def get_dst(split, set_size, mask_type):
    if split == 'train':
        size = 1000
        dst = tf.data.Dataset.from_tensor_slices(tf.range(size))
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, set_size, mask_type], 
            [tf.float32, tf.float32, tf.float32])),
            num_parallel_calls=16)
    else:
        size = 100
        dst = tf.data.Dataset.from_tensor_slices(tf.range(size))
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, set_size, mask_type], 
            [tf.float32, tf.float32, tf.float32])),
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
            dst = dst.prefetch(10)

            dst_it = dst.make_initializable_iterator()
            t, x, b  = dst_it.get_next()
            self.t = tf.reshape(t, [batch_size, set_size, 3])
            self.x = tf.reshape(x, [batch_size, set_size, 3])
            self.b = tf.reshape(b, [batch_size, set_size, 3])
            self.dimension = 3
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        t, x, b = self.sess.run([self.t, self.x, self.b])
        m = np.ones_like(b)
        return {'t':t, 'x':x, 'b':b, 'm':m}