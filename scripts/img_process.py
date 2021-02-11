import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import ast
import logging
import argparse
import pickle
import json
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

data_path = '/data/shapenet/data.pkl'
with open(data_path, 'rb') as f:
    train, valid, test = pickle.load(f)
valid = np.expand_dims(valid, axis=-1)
test = np.expand_dims(test, axis=-1)

# model
model = get_model(params)
model.load()

# run
save_dir = f'{params.exp_dir}/evaluate/set_process/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')
log_file.write(json.dumps(vars(args)))
log_file.write('\n\n')

def evaluate(batch, n_context=None):
    if n_context is not None:
        b = np.zeros_like(batch['b'])
        b[:,:n_context] = 1
        m = np.ones_like(batch['m'])
        batch['b'] = b.copy()
        batch['m'] = m.copy()
    likel, sample = model.execute([model.metric, model.sample], batch)
    mse = np.mean(np.square(sample-batch['x']), axis=(1,2,3,4))
    
    return -likel, mse

def get_angle():
    seen_angles = [i*10 for i in range(36)]
    unseen_angles = [i*10+5 for i in range(36)]
    angles = seen_angles + unseen_angles
    radians = [np.deg2rad(a) for a in angles]
    t = np.stack([np.sin(radians), np.cos(radians)], axis=-1)

    return t.astype(np.float32)

def visualize(batch, real, prefix, n_context=None):
    if n_context is not None:
        x = batch['x']
        b = np.zeros_like(batch['b'])
        b[:,:n_context] = 1
        m = np.ones_like(batch['m'])
        all_t = get_angle()
        K = params.set_size - n_context
        num_sets = len(all_t) // K
        context = x[:,:n_context]
        samples = []
        for i in range(num_sets):
            t = batch['t']
            tx = np.expand_dims(all_t[i*K:(i+1)*K], axis=0)
            t[:,n_context:] = np.repeat(tx, t.shape[0], axis=0)
            batch = {'t':t, 'x':x, 'b':b, 'm':m}
            s = model.execute(model.sample, batch)
            samples.append(s[:,n_context:])
        samples = np.concatenate(samples, axis=1)
        real = np.concatenate([real, np.zeros_like(real)], axis=1)
    else:
        context = batch['x'][:,:n_context]
        s = model.execute(model.sample, batch)
        samples = s[:,n_context:]
        real = batch['x'][:,n_context:]

    with open(f'{prefix}.pkl', 'wb') as f:
        pickle.dump((context, samples, real), f)

    # visualize
    B,N,H,W,C = context.shape
    B,K,H,W,C = samples.shape
    B,R,H,W,C = real.shape
    assert R == K
    for i in range(B):
        xc, xs, xt = context[i], samples[i], real[i]
        xe = np.zeros_like(xc)
        xu = np.concatenate([xe, xt], axis=0)
        xd = np.concatenate([xc, xs], axis=0)
        x = np.stack([xu, xd], axis=0) # [2,N,H,W,C]
        im = np.transpose(x, [0,2,1,3,4]).reshape([2*H,W*(N+K),C]).squeeze()
        plt.imsave(f'{prefix}_{i}.png', im)

# valid
valid_nll = []
valid_mse = []
num_batches = len(validset)
for i in range(num_batches):
    res = evaluate(validset[i], args.n_context)
    valid_nll.append(res[0])
    valid_mse.append(res[1])
valid_nll = np.mean(np.concatenate(valid_nll))
valid_mse = np.mean(np.concatenate(valid_mse))
valid_psnr = 20 * np.log10(255.) - 10 * np.log10(valid_mse)
log_file.write(f'valid_nll: {valid_nll}\n')
log_file.write(f'valid_mse: {valid_mse}\n')
log_file.write(f'valid_psnr: {valid_psnr}\n')

# save_path = f'{save_dir}/valid/'
# os.makedirs(save_path, exist_ok=True)
# for i in range(10):
#     prefix = f'{save_path}/batch_{i}'
#     real = valid[i*params.batch_size:(i+1)*params.batch_size]
#     visualize(validset[i], real, prefix)

# save_path = f'{save_dir}/valid_ctx{args.n_context}/'
# os.makedirs(save_path, exist_ok=True)
# for i in range(0, num_batches, 5):
#     prefix = f'{save_path}/batch_{i}'
#     real = valid[i*params.batch_size:(i+1)*params.batch_size]
#     visualize(validset[i], real, prefix, args.n_context)

# test
test_nll = []
test_mse = []
num_batches = len(testset)
for i in range(num_batches):
    res = evaluate(testset[i], args.n_context)
    test_nll.append(res[0])
    test_mse.append(res[1])
test_nll = np.mean(np.concatenate(test_nll))
test_mse = np.mean(np.concatenate(test_mse))
test_psnr = 20 * np.log10(255.) - 10 * np.log10(test_mse)
log_file.write(f'test_nll: {test_nll}\n')
log_file.write(f'test_mse: {test_mse}\n')
log_file.write(f'test_psnr: {test_psnr}\n')

# save_path = f'{save_dir}/test/'
# os.makedirs(save_path, exist_ok=True)
# for i in range(10):
#     prefix = f'{save_path}/batch_{i}'
#     real = test[i*params.batch_size:(i+1)*params.batch_size]
#     visualize(testset[i], real, prefix)

# save_path = f'{save_dir}/test_ctx{args.n_context}/'
# os.makedirs(save_path, exist_ok=True)
# for i in range(0, num_batches, 5):
#     prefix = f'{save_path}/batch_{i}'
#     real = test[i*params.batch_size:(i+1)*params.batch_size]
#     visualize(testset[i], real, prefix, args.n_context)

log_file.close()