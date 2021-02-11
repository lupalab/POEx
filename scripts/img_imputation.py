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
save_dir = f'{params.exp_dir}/evaluate/set_imputation/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')

def evaluate(batch):
    average_mse = []
    for _ in range(5):
        sample = model.execute(model.sample, batch)
        mse = np.mean(np.square(sample-batch['x']), axis=(1,2,3,4))
        average_mse.append(mse)
    average_mse = np.mean(np.stack(average_mse, axis=0), axis=0)

    return average_mse

def visualize(batch, prefix):
    s = model.execute(model.sample, batch)
    x, b, m = batch['x'], batch['b'], batch['m']
    B,N,H,W,C = s.shape
    for i in range(B):
        ss, xx, bb, mm = s[i], x[i], b[i], m[i]
        ss = np.transpose(ss, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        xx = np.transpose(xx, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        bb = np.transpose(bb, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        mm = np.transpose(mm, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        xm = xx * mm + (1-mm) * 128
        xo = xx * bb + (1-bb) * 128
        img = np.concatenate([xm, xo, ss]).astype(np.uint8)

        plt.imsave(f'{prefix}_{i}.png', img) 


# valid
save_path = f'{save_dir}/valid/'
os.makedirs(save_path, exist_ok=True)

valid_mse = []
num_batches = len(validset)
for i in range(num_batches):
    valid_mse.append(evaluate(validset[i]))
valid_mse = np.mean(np.concatenate(valid_mse))
valid_psnr = 20 * np.log10(255.) - 10 * np.log10(valid_mse)
log_file.write(f'valid_mse: {valid_mse}\n')
log_file.write(f'valid_psnr: {valid_psnr}\n')

for i in range(10):
    prefix = f'{save_path}/batch_{i}'
    visualize(validset[i], prefix)

# test
save_path = f'{save_dir}/test/'
os.makedirs(save_path, exist_ok=True)

test_mse = []
num_batches = len(testset)
for i in range(num_batches):
    test_mse.append(evaluate(testset[i]))
test_mse = np.mean(np.concatenate(test_mse))
test_psnr = 20 * np.log10(255.) - 10 * np.log10(test_mse)
log_file.write(f'test_mse: {test_mse}\n')
log_file.write(f'test_psnr: {test_psnr}\n')

for i in range(10):
    prefix = f'{save_path}/batch_{i}'
    visualize(testset[i], prefix)

log_file.close()