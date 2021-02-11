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
save_dir = f'{params.exp_dir}/evaluate/set_imputation/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')

def to_image(x):
    x = x[...,0] + x[...,1] * 1j
    x = np.fft.ifftshift(x, axes=(-2,-1))
    x = np.fft.ifft2(x)
    x = np.absolute(x)
    return x

def evaluate(batch):
    sample = model.execute(model.sample, batch)
    s = to_image(sample)
    x = to_image(batch['x'])
    mse = np.mean(np.square(s-x), axis=(1,2,3))

    return mse

def visualize(batch, prefix):
    sample = model.execute(model.sample, batch)
    s = to_image(sample)
    x = to_image(batch['x'])
    b = batch['b'][...,0]
    B,N,H,W = x.shape
    for i in range(B):
        ss, xx, bb = s[i], x[i], b[i]
        ss = np.transpose(ss, [1,0,2]).reshape(H,W*N)
        xx = np.transpose(xx, [1,0,2]).reshape(H,W*N)
        bb = np.transpose(bb, [1,0,2]).reshape(H,W*N)
        img = np.concatenate([bb, ss, xx]).astype(np.float32)

        plt.imsave(f'{prefix}_{i}.png', img) 

# test
save_path = f'{save_dir}/test/'
os.makedirs(save_path, exist_ok=True)

test_mse = []
num_batches = len(testset)
for i in range(num_batches):
    test_mse.append(evaluate(testset[i]))
test_mse = np.mean(np.concatenate(test_mse))
test_psnr = 20 * np.log10(1.) - 10 * np.log10(test_mse)
log_file.write(f'test_mse: {test_mse}\n')
log_file.write(f'test_psnr: {test_psnr}\n')

for i in range(10):
    prefix = f'{save_path}/batch_{i}'
    visualize(testset[i], prefix)

log_file.close()