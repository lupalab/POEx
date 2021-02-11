import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import ast
import logging
import argparse
import numpy as np
import tensorflow as tf
from pprint import pformat
import matplotlib.pyplot as plt

from utils.hparams import HParams
from models import get_model
from datasets import get_cached_data

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--n_context', type=int, default=50)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# data
testset = get_cached_data(params, 'test')

# model
model = get_model(params)
model.load()

# run
save_dir = f'{params.exp_dir}/evaluate/imfns_imputation/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')

def evaluate(batch):
    log_likel = model.execute(model.metric, batch)

    return -log_likel

def visualize(batch, prefix, n_context=None):
    if n_context is not None:
        index, image = batch['xt'], batch['yt']
        N = index.shape[2]
        idx = np.random.choice(N, n_context)
        x_ctx = index[:,:,idx]
        y_ctx = image[:,:,idx]
        batch['idx'] = idx
        batch['xc'] = x_ctx
        batch['yc'] = y_ctx
        batch['xt'] = index
        batch['yt'] = image
    m = model.execute(model.mean, batch)
    B,K,N,C = m.shape
    idx, xc, yc, xt, yt = batch['idx'], batch['xc'], batch['yc'], batch['xt'], batch['yt']
    yo = np.ones_like(yt) * 128
    yo[:,:,idx] = (yc + 0.5) * 255.
    yt =  (yt + 0.5) * 255.
    m = (m + 0.5) * 255.
    for i in range(B):
        yoi, yti, mi = yo[i], yt[i], m[i]
        yoi = np.reshape(yoi, [K,28,28]).astype(np.uint8)
        yoi = np.reshape(np.transpose(yoi, [1,0,2]), [28, K*28])
        yti = np.reshape(yti, [K,28,28]).astype(np.uint8)
        yti = np.reshape(np.transpose(yti, [1,0,2]), [28, K*28])
        mi = np.reshape(mi, [K,28,28]).astype(np.uint8)
        mi = np.reshape(np.transpose(mi, [1,0,2]), [28, K*28])
        img = np.concatenate([yoi, mi, yti], axis=0)

        plt.imsave(f'{prefix}_{i}.png', img)

# test
test_nll = []
num_batches = len(testset)
for i in range(num_batches):
    test_nll.append(evaluate(testset[i]))
test_nll = np.mean(np.concatenate(test_nll))
log_file.write(f'test_nll: {test_nll}\n')

save_path = f'{save_dir}/test/'
os.makedirs(save_path, exist_ok=True)
for i in range(10):
    prefix = f'{save_path}/batch_{i}'
    visualize(testset[i], prefix)

save_path = f'{save_dir}/test_ctx{args.n_context}'
os.makedirs(save_path, exist_ok=True)
for i in range(10):
    prefix = f'{save_path}/batch_{i}'
    visualize(testset[i], prefix, args.n_context)

log_file.close()