import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import ast
import logging
import argparse
import pickle
import numpy as np
import tensorflow as tf
from pprint import pformat
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.hparams import HParams
from models import get_model
from datasets import get_cached_data

from models.pc_metrics.metrics import distChamfer, distEMD

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--n_context', type=int, default=256)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)
# modify config
params.mask_type = 'det_expand'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# data
validset = get_cached_data(params, 'valid')
testset = get_cached_data(params, 'test')

# model
model = get_model(params)
model.load()

# run
save_dir = f'{params.exp_dir}/evaluate/set_expansion/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')

def evaluate(batch):
    sample = model.execute(model.sample, batch)
    chd = distChamfer(sample, batch['x'])
    emd = distEMD(sample, batch['x'])

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

# valid
save_path = f'{save_dir}/valid/'
os.makedirs(save_path, exist_ok=True)

valid_chd = []
valid_emd = []
num_batches = len(validset)
for i in range(num_batches):
    res = evaluate(validset[i])
    valid_chd.append(res[0])
    valid_emd.append(res[1])
valid_chd = np.mean(np.concatenate(valid_chd))
valid_emd = np.mean(np.concatenate(valid_emd))
log_file.write(f'valid_chd: {valid_chd}\n')
log_file.write(f'valid_emd: {valid_emd}\n')

for i in range(0, num_batches, 1):
    prefix = f'{save_path}/batch_{i}'
    visualize(validset[i], prefix)

# test
save_path = f'{save_dir}/test/'
os.makedirs(save_path, exist_ok=True)

test_chd = []
test_emd = []
num_batches = len(testset)
for i in range(num_batches):
    res = evaluate(testset[i])
    test_chd.append(res[0])
    test_emd.append(res[1])
test_chd = np.mean(np.concatenate(test_chd))
test_emd = np.mean(np.concatenate(test_emd))
log_file.write(f'test_chd: {test_chd}\n')
log_file.write(f'test_emd: {test_emd}\n')

for i in range(0, num_batches, 1):
    prefix = f'{save_path}/batch_{i}'
    visualize(testset[i], prefix)

log_file.close()