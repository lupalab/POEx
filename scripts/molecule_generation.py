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
from collections import defaultdict

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
trainset = get_cached_data(params, 'train')
atoms = np.array([1,6,7,8,9])

# model
model = get_model(params)
model.load()

# run
save_dir = f'{params.exp_dir}/evaluate/set_generation/'
os.makedirs(save_dir, exist_ok=True)

# train
save_path = f'{save_dir}/train/'
os.makedirs(save_path, exist_ok=True)

res_p = defaultdict(list)
res_t = defaultdict(list)
num_batches = len(trainset)
for i in range(num_batches):
    batch = trainset[i]
    sample = model.execute(model.sample, batch)
    num_atoms = sample.shape[1]
    positions = sample #[B,N,3]
    atom_type = np.sum(batch['t'] * atoms, axis=-1).astype(np.int32) #[B,N]
    res_p[num_atoms].append(positions)
    res_t[num_atoms].append(atom_type)

print(res_p.keys())

results = {}
for k in res_p.keys():
    results[k] = {}
    p, t = res_p[k], res_t[k]
    results[k]['_positions'] = np.concatenate(p, axis=0)
    results[k]['_atomic_numbers'] = np.concatenate(t, axis=0)

with open(f'{save_path}/generated.mol_dict', 'wb') as f:
    pickle.dump(results, f)
