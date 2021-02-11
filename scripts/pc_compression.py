import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import ast
import h5py
import logging
import argparse
import pickle
import numpy as np
import tensorflow as tf
from pprint import pformat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.special import softmax

from utils.hparams import HParams
from models import get_model
from datasets import get_cached_data

from models.pc_metrics.metrics import distChamfer, distEMD

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--biased', action='store_true')
parser.add_argument('--n_points', type=int, default=128)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)
# modify config
params.dataset = 'modelnet'
params.mask_type = 'det_expand'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# data
if not args.biased:
    validset = get_cached_data(params, 'valid')
elif os.path.isfile('.modelnet_airplane_biased'):
    with open('.modelnet_airplane_biased', 'rb') as f:
        validset = pickle.load(f)
else:
    with h5py.File('/data/pointcloud/ModelNet40_cloud.h5', 'r') as f:
        train_cloud = np.array(f['tr_cloud'])
        train_labels = np.array(f['tr_labels'])
        test_cloud = np.array(f['test_cloud'])
        test_labels = np.array(f['test_labels'])
    inds = np.where(test_labels == 0)[0]
    pcs = []
    pcs_org = []
    for i in inds:
        x = test_cloud[i].astype(np.float32)
        # preprocess
        x_max = np.max(x, axis=0)
        x_min = np.min(x, axis=0)
        x = (x - x_min) / (x_max - x_min)
        x -= np.mean(x, axis=0)
        # subsample balanced
        ind = np.random.choice(x.shape[0], 2048, replace=False)
        x_org = x[ind]
        pcs_org.append(x_org)
        # subsample
        ind1 = np.random.choice(x.shape[0], 256, replace=False)
        x1 = x[ind1]
        dist = np.sum(np.square(x), axis=1)
        T = 0.01
        p = softmax(-dist/T)
        ind2 = np.random.choice(x.shape[0], 2048-256, replace=False, p=p)
        x2 = x[ind2]
        x = np.concatenate([x1, x2], axis=0)
        pcs.append(x)

    validset = []
    bs = params.batch_size
    num_batches = len(pcs) // bs
    for i in range(num_batches):
        xs_org = np.stack(pcs_org[i*bs:(i+1)*bs])
        xs = np.stack(pcs[i*bs:(i+1)*bs])
        batch = {}
        batch['x_org'] = xs_org
        batch['x'] = xs
        batch['b'] = np.zeros_like(xs)
        batch['m'] = np.ones_like(xs)
        validset.append(batch)

    with open('.modelnet_airplane_biased', 'wb') as f:
        pickle.dump(validset, f)

# model
model = get_model(params)
model.load()

# run
save_dir = f'{params.exp_dir}/evaluate/set_compression/'
if args.biased:
    save_dir = f'{params.exp_dir}/evaluate/set_compression_biased/'
os.makedirs(save_dir, exist_ok=True)

def compress_pseudo(batch):
    B = batch['x'].shape[0]
    b = np.zeros_like(batch['x'])
    for step in range(args.n_points):
        batch['b'] = b.copy()
        set_ll = model.execute(model.log_likel, batch)
        set_ll[b[:,:,0]==1] = np.inf
        inds = np.argmin(set_ll, axis=1)
        b[range(B), inds] = 1.0
    assert np.all(np.sum(b[:,:,0], axis=1) == args.n_points)

    return b

def compress_active(batch):
    B = batch['x'].shape[0]
    b = np.zeros_like(batch['x'])
    for step in range(args.n_points):
        batch['b'] = b.copy()
        set_ll = model.execute(model.set_metric, batch)
        set_ll[b[:,:,0]==1] = np.inf
        inds = np.argmin(set_ll, axis=1)
        b[range(B), inds] = 1.0
    assert np.all(np.sum(b[:,:,0], axis=1) == args.n_points)

    return b

def compress_uniform(batch):
    b = np.zeros_like(batch['x'])
    inds = np.random.choice(b.shape[1], args.n_points, replace=False)
    b[:,inds] = 1.0

    return b

def compress_kmeans(batch):
    x = batch['x']
    B = x.shape[0]
    b = np.zeros_like(batch['x'])
    for i in range(B):
        xx = x[i]
        kmeans = KMeans(n_clusters=args.n_points, n_jobs=-1).fit(xx)
        center = kmeans.cluster_centers_
        dist = np.sum(np.square(np.expand_dims(xx, axis=1) - center), axis=-1)
        inds = np.argmin(dist, axis=0)
        b[i, inds] = 1.0

    return b

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = np.sum(np.square(points1), axis=-1).reshape([B, M, 1]) + \
            np.sum(np.square(points2), axis=-1).reshape([B, 1, N])
    dists -= 2 * np.matmul(points1, np.transpose(points2, [0, 2, 1]))
    mask = (dists < 0).astype(np.float32)
    dists = mask * np.ones_like(dists) * 1e-7 + (1-mask) * dists # Very Important for dist = 0.
    return np.sqrt(dists)

def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    B, N, C = xyz.shape
    centroids = np.zeros((B, M), dtype=np.int32)
    dists = np.ones((B, N)) * 1e5
    inds = np.random.randint(0, N, size=(B, ), dtype=np.int32)
    batchlists = np.arange(0, B, dtype=np.int32)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = np.squeeze(get_dists(np.expand_dims(cur_point, 1), xyz))
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = np.argmax(dists, axis=1)
    return centroids

def compress_farthest(batch):
    x = batch['x']
    B = x.shape[0]
    b = np.zeros_like(batch['x'])
    inds = fps(x, args.n_points) # [B,M]
    for i in range(B):
        b[i, inds[i]] = 1

    return b

def evaluate(batch):
    sample = model.execute(model.sample, batch)
    data = batch['x_org'] if args.biased else batch['x']
    chd = distChamfer(sample, data)
    emd = distEMD(sample, data)

    return chd, emd

def visualize(batch, prefix):
    sample = model.execute(model.sample, batch)
    
    with open(f'{prefix}.pkl', 'wb') as f:
        pickle.dump((batch, sample), f)
    
    for i in range(sample.shape[0]):
        s = sample[i]
        x = batch['x'][i]
        o = np.where(batch['b'][i,:,0]==1)[0]
        xo = x[o]
        fig = plt.figure(figsize=(7.5,2.5))
        ax = fig.add_subplot(131, projection='3d')
        ax.scatter(xo[:,0], xo[:,1], xo[:,2], c='g', s=5)
        ax.axis('off')
        ax.grid(False)
        ax = fig.add_subplot(132, projection='3d')
        ax.scatter(s[:,0], s[:,1], s[:,2], c='g', s=5)
        ax.axis('off')
        ax.grid(False)
        ax = fig.add_subplot(133, projection='3d')
        ax.scatter(x[:,0], x[:,1], x[:,2], c='g', s=5)
        ax.axis('off')
        ax.grid(False)
        plt.savefig(f'{prefix}_{i}.png')
        plt.close('all')

# pseudo 
save_path = f'{save_dir}/valid_p{args.n_points}_pseudo/'
os.makedirs(save_path, exist_ok=True)
log_file = open(f'{save_path}/log.txt', 'w')

valid_chd = []
valid_emd = []
batches = []
num_batches = len(validset)
for i in range(num_batches):
    prefix = f'{save_path}/batch_{i}'
    batch = validset[i]
    b = compress_pseudo(batch)
    batch['b'] = b.copy()
    batches.append(batch)
    res = evaluate(batch)
    valid_chd.append(res[0])
    valid_emd.append(res[1])
    visualize(batch, prefix)
valid_chd = np.mean(np.concatenate(valid_chd))
valid_emd = np.mean(np.concatenate(valid_emd))
log_file.write(f'valid_chd: {valid_chd}\n')
log_file.write(f'valid_emd: {valid_emd}\n')

with open(f'{save_path}/results.pkl', 'wb') as f:
    pickle.dump(batches, f)

log_file.close()

# # active 
# save_path = f'{save_dir}/valid_p{args.n_points}_active/'
# os.makedirs(save_path, exist_ok=True)
# log_file = open(f'{save_path}/log.txt', 'w')

# valid_chd = []
# valid_emd = []
# num_batches = len(validset)
# for i in range(num_batches):
#     prefix = f'{save_path}/batch_{i}'
#     batch = validset[i]
#     b = compress_active(batch)
#     batch['b'] = b.copy()
#     res = evaluate(batch)
#     valid_chd.append(res[0])
#     valid_emd.append(res[1])
#     visualize(batch, prefix)
# valid_chd = np.mean(np.concatenate(valid_chd))
# valid_emd = np.mean(np.concatenate(valid_emd))
# log_file.write(f'valid_chd: {valid_chd}\n')
# log_file.write(f'valid_emd: {valid_emd}\n')

# log_file.close()

# uniform
save_path = f'{save_dir}/valid_p{args.n_points}_uniform/'
os.makedirs(save_path, exist_ok=True)
log_file = open(f'{save_path}/log.txt', 'w')

valid_chd = []
valid_emd = []
batches = []
num_batches = len(validset)
for i in range(num_batches):
    prefix = f'{save_path}/batch_{i}'
    batch = validset[i]
    b = compress_uniform(batch)
    batch['b'] = b.copy()
    batches.append(batch)
    res = evaluate(batch)
    valid_chd.append(res[0])
    valid_emd.append(res[1])
    visualize(batch, prefix)
valid_chd = np.mean(np.concatenate(valid_chd))
valid_emd = np.mean(np.concatenate(valid_emd))
log_file.write(f'valid_chd: {valid_chd}\n')
log_file.write(f'valid_emd: {valid_emd}\n')

with open(f'{save_path}/results.pkl', 'wb') as f:
    pickle.dump(batches, f)

log_file.close()

# kmeans
save_path = f'{save_dir}/valid_p{args.n_points}_kmeans/'
os.makedirs(save_path, exist_ok=True)
log_file = open(f'{save_path}/log.txt', 'w')

valid_chd = []
valid_emd = []
batches = []
num_batches = len(validset)
for i in range(num_batches):
    prefix = f'{save_path}/batch_{i}'
    batch = validset[i]
    b = compress_kmeans(batch)
    batch['b'] = b.copy()
    batches.append(batch)
    res = evaluate(batch)
    valid_chd.append(res[0])
    valid_emd.append(res[1])
    visualize(batch, prefix)
valid_chd = np.mean(np.concatenate(valid_chd))
valid_emd = np.mean(np.concatenate(valid_emd))
log_file.write(f'valid_chd: {valid_chd}\n')
log_file.write(f'valid_emd: {valid_emd}\n')

with open(f'{save_path}/results.pkl', 'wb') as f:
    pickle.dump(batches, f)

log_file.close()

# farthest
save_path = f'{save_dir}/valid_p{args.n_points}_farthest/'
os.makedirs(save_path, exist_ok=True)
log_file = open(f'{save_path}/log.txt', 'w')

valid_chd = []
valid_emd = []
batches = []
num_batches = len(validset)
for i in range(num_batches):
    prefix = f'{save_path}/batch_{i}'
    batch = validset[i]
    b = compress_farthest(batch)
    batch['b'] = b.copy()
    batches.append(batch)
    res = evaluate(batch)
    valid_chd.append(res[0])
    valid_emd.append(res[1])
    visualize(batch, prefix)
valid_chd = np.mean(np.concatenate(valid_chd))
valid_emd = np.mean(np.concatenate(valid_emd))
log_file.write(f'valid_chd: {valid_chd}\n')
log_file.write(f'valid_emd: {valid_emd}\n')

with open(f'{save_path}/results.pkl', 'wb') as f:
    pickle.dump(batches, f)

log_file.close()