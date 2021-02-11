import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import ast
import logging
import argparse
import pickle
import numpy as np
from glob import glob
import tensorflow as tf
from pprint import pformat
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.hparams import HParams
from models import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# data
data_path = '/data/Colon/goodmesh/'
file_list = glob(f'{data_path}/*/*.obj')
meshes = []
for fname in file_list:
    mesh = o3d.io.read_triangle_mesh(fname)
    meshes.append(mesh)
train_meshes = meshes[:-5]
test_meshes = meshes[-5:]

# model
model = get_model(params)
model.load()

# run
save_dir = f'{params.exp_dir}/evaluate/colon_imputation/'
os.makedirs(save_dir, exist_ok=True)

def visualize(mesh, data, sample, mask, prefix):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(sample)
    o_color = np.array((0.0, 1.0, 0.0))
    u_color = np.array((0.0, 0.0, 1.0))
    color = mask * o_color + (1-mask) * u_color
    pc.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(f'{prefix}.pts', pc)
    o3d.io.write_triangle_mesh(f'{prefix}.off', mesh)

    # save screen shot
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(mesh)
    # vis.add_geometry(pc)
    # vis.run()
    # image = vis.capture_screen_float_buffer()
    # plt.imsave(f'{prefix}.png', np.asarray(image))
    # vis.destroy_window()

# test
save_path = f'{save_dir}/test/'
os.makedirs(save_path, exist_ok=True)

def get_coord(x, num_blocks):
    assert np.max(x) <= 1.0
    coord = np.floor(x * num_blocks) / num_blocks

    return coord

def generate_mask(data):
    radius = np.random.uniform(0.2)
    coord = get_coord(data, 10)
    anchor = coord[np.random.choice(len(data))]
    dist = np.sqrt(np.sum(np.square(coord - anchor), axis=1))
    m = (dist > radius).astype(np.float32)
    m = np.repeat(np.expand_dims(m, axis=1), 3, axis=1)

    return m

for i in range(100):
    idx = i % len(test_meshes)
    mesh = test_meshes[idx]
    pcd = mesh.sample_points_uniformly(number_of_points=2048)
    x = np.asarray(pcd.points, dtype=np.float32)
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x - x_min) / (x_max - x_min)
    offset = np.mean(x, axis=0)
    x = x - offset
    b = generate_mask(x)
    m = np.ones_like(b)
    t = get_coord(x, 10)
    batch = {'t': np.expand_dims(t, axis=0), 
             'x': np.expand_dims(x, axis=0),
             'b': np.expand_dims(b, axis=0), 
             'm': np.expand_dims(m, axis=0)}
    sample = model.execute(model.sample, batch)
    sample = (sample[0] + offset) * (x_max - x_min) + x_min
    data = (x + offset) * (x_max - x_min) + x_min
    mask = b
    prefix = f'{save_path}/mesh_{i}'
    visualize(mesh, data, sample, mask, prefix)

