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
import matplotlib.pyplot as plt

from utils.hparams import HParams
from models import get_model
from datasets import get_cached_data

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--n_context', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)

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

def evaluate(batch, n_context):
    x = batch['x']
    b = np.zeros_like(batch['b'])
    b[:,:n_context] = 1
    m = np.ones_like(batch['m'])
    batch = {'x':x, 'b':b, 'm':m}
    likel = model.execute(model.metric, batch)
    
    return -likel

def visualize(batch, prefix, n_context):
    x = batch['x']
    b = np.zeros_like(batch['b'])
    b[:,:n_context] = 1
    m = np.ones_like(batch['m'])
    batch = {'x':x, 'b':b, 'm':m}
    sample = model.execute(model.sample, batch)

    with open(f'{prefix}.pkl', 'wb') as f:
        pickle.dump((batch, sample), f)

    # visualize
    B,N,H,W,C = sample.shape
    for i in range(B):
        x = sample[i]
        im = np.transpose(x, [1,0,2,3]).reshape([H,W*N,C]).squeeze()
        plt.imsave(f'{prefix}_{i}.png', im)

# valid
save_path = f'{save_dir}/valid_ctx{args.n_context}/'
os.makedirs(save_path, exist_ok=True)
for i in range(10):
    prefix = f'{save_path}/batch_{i}'
    visualize(validset[i], prefix, args.n_context)

# test
save_path = f'{save_dir}/test_ctx{args.n_context}/'
os.makedirs(save_path, exist_ok=True)
for i in range(10):
    prefix = f'{save_path}/batch_{i}'
    visualize(testset[i], prefix, args.n_context)

log_file.close()