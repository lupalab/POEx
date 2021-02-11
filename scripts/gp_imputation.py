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
save_dir = f'{params.exp_dir}/evaluate/gp_imputation/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')

def evaluate(batch):
    log_likel = model.execute(model.metric, batch)

    return -log_likel

def visualize(batch, prefix):
    m, s = model.execute([model.mean, model.std], batch)

    with open(f'{prefix}.pkl', 'wb') as f:
        pickle.dump((batch, m, s), f)

    B,K,N,C = m.shape
    xc, yc, xt, yt = batch['xc'], batch['yc'], batch['xt'], batch['yt']
    for i in range(B):
        mm, ss, xxc, yyc, xxt, yyt = m[i,:,:,0], s[i,:,:,0], xc[i,:,:,0], yc[i,:,:,0], xt[i,:,:,0], yt[i,:,:,0]
        fig = plt.figure(figsize=(4.0, 2.5*K))
        for k in range(K):
            idx = np.argsort(xxt[k])
            ax = fig.add_subplot(K,1,k+1)
            ax.plot(xxc[k], yyc[k], 'rx', markersize=8)
            ax.plot(xxt[k], yyt[k], 'ko', markersize=3)
            ax.plot(xxt[k,idx], mm[k,idx], 'b', linewidth=2)
            plt.fill_between(
                xxt[k,idx],
                mm[k,idx] - ss[k,idx],
                mm[k,idx] + ss[k,idx],
                alpha=0.2,
                facecolor='#65c9f7',
                interpolate=True)
        plt.savefig(f'{prefix}_{i}.png')
        plt.close('all')

# test
save_path = f'{save_dir}/test/'
os.makedirs(save_path, exist_ok=True)

test_nll = []
num_batches = len(testset)
for i in range(num_batches):
    test_nll.append(evaluate(testset[i]))
test_nll = np.mean(np.concatenate(test_nll))
log_file.write(f'test_nll: {test_nll}\n')

for i in range(10):
    prefix = f'{save_path}/batch_{i}'
    visualize(testset[i], prefix)

log_file.close()