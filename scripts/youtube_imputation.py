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

from models.img_metrics.ssim import ssim_np

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
save_dir = f'{params.exp_dir}/evaluate/video_imputation/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')

def batch_ssim(x, s):
    average_ssim = []
    B,N,H,W,C = x.shape
    x = np.reshape(x, [B*N,H,W,C])
    s = np.reshape(s, [B*N,H,W,C])
    res = ssim_np(x, s, size_average=False)
    average_ssim = res.reshape([B,N]).mean(axis=1)

    return average_ssim

def evaluate(batch):
    average_mse = []
    average_ssim = []
    for _ in range(5):
        sample = model.execute(model.sample, batch)
        mse = np.sum(np.square(sample-batch['x'])*batch['m']*(1-batch['b']), axis=(2,3,4))
        num = np.sum(batch['m']*(1-batch['b']), axis=(2,3,4))
        num = np.maximum(num, np.ones_like(num))
        mse = np.mean(mse/num, axis=1)
        average_mse.append(mse)
        average_ssim.append(batch_ssim(batch['x'], sample))
    average_mse = np.mean(np.stack(average_mse, axis=0), axis=0)
    average_ssim = np.mean(np.stack(average_ssim, axis=0), axis=0)

    return average_mse, average_ssim

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

# test
save_path = f'{save_dir}/test/'
os.makedirs(save_path, exist_ok=True)

test_mse = []
test_ssim = []
num_batches = len(testset)
for i in range(num_batches):
    batch = testset[i]
    res = evaluate(batch)
    test_mse.append(res[0])
    test_ssim.append(res[1])
test_mse = np.mean(np.concatenate(test_mse))
test_psnr = 20 * np.log10(255.) - 10 * np.log10(test_mse)
test_ssim = np.mean(np.concatenate(test_ssim))
log_file.write(f'test_mse: {test_mse}\n')
log_file.write(f'test_psnr: {test_psnr}\n')
log_file.write(f'test_ssim: {test_ssim}\n')

for i in range(10):
    prefix = f'{save_path}/batch_{i}'
    visualize(testset[i], prefix)

log_file.close()