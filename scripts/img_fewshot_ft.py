import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import ast
import copy
import json
import tqdm
import logging
import argparse
import pickle
import numpy as np
import tensorflow as tf
from pprint import pformat
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils.hparams import HParams
from models import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--N', type=int, default=5)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--Q', type=int, default=15)
parser.add_argument('--G', type=int, default=100)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(args.seed)
tf.set_random_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = f'cuda:{args.gpu}'

# model
model = get_model(params)
model.load()

# load data
data_path = '/data/omniglot/'
with open(data_path + 'train_vinyals_aug90.pkl', 'rb') as f:
    train_dict = pickle.load(f, encoding='bytes')
    num_train_labels = len(train_dict[b'label_str'])
with open(data_path + 'val_vinyals_aug90.pkl', 'rb') as f:
    valid_dict = pickle.load(f, encoding='bytes')
    num_valid_labels = len(valid_dict[b'label_str'])
with open(data_path + 'test_vinyals_aug90.pkl', 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')
    num_test_labels = len(test_dict[b'label_str'])

# few shot dataset
def _parse_train(i, N, K, Q):
    label_ids = np.random.choice(num_train_labels, N, replace=False)
    data = []
    label = []
    query = []
    target = []
    for k, lab in enumerate(label_ids):
        inds = np.where(train_dict[b'labels']==lab)[0]
        ind = np.random.choice(inds, K, replace=False)
        image = train_dict[b'images'][ind] #[K,28,28,1]
        clab = np.ones([K], dtype=np.int64) * k
        data.append(image)
        label.append(clab)

        inds = list(set(inds) - set(ind))
        ind = np.random.choice(inds, Q, replace=False)
        image = train_dict[b'images'][ind] #[Q,28,28,1]
        clab = np.ones([Q], dtype=np.int64) * k
        query.append(image)
        target.append(clab)
    data = np.stack(data) #[N,K,28,28,1]
    data = np.pad(data, ((0,0),(0,0),(2,2),(2,2),(0,0)), mode='constant')
    label = np.stack(label) #[N,K]
    
    query = np.stack(query) #[N,Q,28,28,1]
    query = np.pad(query, ((0,0),(0,0),(2,2),(2,2),(0,0)), mode='constant')
    target = np.stack(target) #[N,Q]

    return data, label, query, target

def _parse_valid(i, N, K, Q):
    label_ids = np.random.choice(num_valid_labels, N, replace=False)
    data = []
    label = []
    query = []
    target = []
    for k, lab in enumerate(label_ids):
        inds = np.where(valid_dict[b'labels']==lab)[0]
        ind = np.random.choice(inds, K, replace=False)
        image = valid_dict[b'images'][ind] #[K,28,28,1]
        clab = np.ones([K], dtype=np.int64) * k
        data.append(image)
        label.append(clab)

        inds = list(set(inds) - set(ind))
        ind = np.random.choice(inds, Q, replace=False)
        image = train_dict[b'images'][ind] #[Q,28,28,1]
        clab = np.ones([Q], dtype=np.int64) * k
        query.append(image)
        target.append(clab)
    data = np.stack(data) #[N,K,28,28,1]
    data = np.pad(data, ((0,0),(0,0),(2,2),(2,2),(0,0)), mode='constant')
    label = np.stack(label) #[N,K]

    query = np.stack(query) #[N,Q,28,28,1]
    query = np.pad(query, ((0,0),(0,0),(2,2),(2,2),(0,0)), mode='constant')
    target = np.stack(target) #[N,Q]

    return data, label, query, target

def _parse_test(i, N, K, Q):
    label_ids = np.random.choice(num_test_labels, N, replace=False)
    data = []
    label = []
    query = []
    target = []
    for k, lab in enumerate(label_ids):
        inds = np.where(test_dict[b'labels']==lab)[0]
        ind = np.random.choice(inds, K, replace=False)
        image = test_dict[b'images'][ind] #[K,28,28,1]
        clab = np.ones([K], dtype=np.int64) * k
        data.append(image)
        label.append(clab)

        inds = list(set(inds) - set(ind))
        ind = np.random.choice(inds, Q, replace=False)
        image = train_dict[b'images'][ind] #[Q,28,28,1]
        clab = np.ones([Q], dtype=np.int64) * k
        query.append(image)
        target.append(clab)
    data = np.stack(data) #[N,K,28,28,1]
    data = np.pad(data, ((0,0),(0,0),(2,2),(2,2),(0,0)), mode='constant')
    label = np.stack(label) #[N,K]

    query = np.stack(query) #[N,Q,28,28,1]
    query = np.pad(query, ((0,0),(0,0),(2,2),(2,2),(0,0)), mode='constant')
    target = np.stack(target) #[N,Q]

    return data, label, query, target

def get_dst(split, N, K, Q):
    if split == 'train':
        size = 10000
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, N, K, Q], 
            [tf.uint8, tf.int64, tf.uint8, tf.int64])),
            num_parallel_calls=16)
    elif split == 'valid':
        size = 600
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i, N, K, Q], 
            [tf.uint8, tf.int64, tf.uint8, tf.int64])),
            num_parallel_calls=16)
    else:
        size = 600
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, N, K, Q], 
            [tf.uint8, tf.int64, tf.uint8, tf.int64])),
            num_parallel_calls=16)

    return dst, size

class Omniglot(object):
    def __init__(self, split, N, K, Q):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            dst, size = get_dst(split, N, K, Q)
            self.size = size
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            self.x, self.y, self.q, self.t = dst_it.get_next()
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next(self):
        x, y, q, t = self.sess.run([self.x, self.y, self.q, self.t])

        return {'x':x, 'y':y, 'q':q, 't':t}

def to_classifier_format(dataset):
    x, y, q, t = dataset['x'], dataset['y'], dataset['q'], dataset['t']
    x = x[:,:,2:-2,2:-2].reshape([x.shape[0] * x.shape[1], 1, 28, 28])
    x = x.astype(np.float32) / 255.
    x = torch.from_numpy(x)
    y = y.reshape([y.shape[0] * y.shape[1]])
    y = torch.from_numpy(y)
    q = q[:,:,2:-2,2:-2].reshape([q.shape[0] * q.shape[1], 1, 28, 28])
    q = q.astype(np.float32) / 255.
    q = torch.from_numpy(q)
    t = t.reshape([t.shape[0] * t.shape[1]])
    t = torch.from_numpy(t)

    return {'x':x, 'y':y, 'q':q, 't':t}

def to_generator_format(dataset):
    x = np.zeros([args.N,params.set_size,32,32,1], dtype=np.uint8)
    x[:,:args.K] = dataset['x'].copy()
    b = np.zeros([args.N,params.set_size,32,32,1], dtype=np.uint8)
    b[:,:args.K] = 1.0
    m = np.ones([args.N,params.set_size,32,32,1], dtype=np.uint8)

    return {'x':x, 'b':b, 'm':m}

def augment(dataset, numG):
    batch = to_generator_format(dataset)
    num_batches = numG // (params.set_size-args.K) + 1
    samples = []
    for _ in range(num_batches):
        s = model.execute(model.sample, batch)
        samples.append(s[:,args.K:])
    samples = np.concatenate(samples, axis=1)[:,:numG]
    labels = np.repeat(dataset['y'][:,:1], numG, axis=1)
    x = np.concatenate([dataset['x'], samples], axis=1)
    y = np.concatenate([dataset['y'], labels], axis=1)
    dataset = copy.deepcopy(dataset)
    dataset['x'] = x.copy()
    dataset['y'] = y.copy()

    return dataset

class Net(nn.Module):
    def __init__(self, dim_in, dim_out_ft, dim_out_fs):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 2, 1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.ReLU(True),
            nn.Flatten()
        )
        self.linear = nn.Linear(64, dim_out_ft)
        self.classifier = nn.Linear(64, dim_out_fs)

    def fix_backbone(self):
        for p in self.net.parameters():
            p.requires_grad_(False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def fewshot(self, x):
        return self.classifier(self.net(x))

    def finetune(self, x):
        return self.linear(self.net(x))

class Runner(object):
    def __init__(self):
        self.trainset = Omniglot('train', args.N, args.K, args.Q)
        self.testset = Omniglot('test', args.N, args.K, args.Q)
        self.net = Net(1, num_train_labels, args.N).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def initialize(self, state_dict=None):
        self.net._initialize_weights()
        if state_dict is not None:
            self.net.load_state_dict(state_dict)
            self.net.fix_backbone()
        parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        self.optimizer = optim.SGD(parameters, lr=0.001)

    def pretrain(self, data, label):
        data = data.to(device)
        label = label.to(device)
        self.net.train()
        pbar = tqdm.trange(5000)
        for i in pbar:
            ind = torch.randint(0, data.shape[0], size=(1000,))
            x, y = data[ind], label[ind]
            self.optimizer.zero_grad()
            prob = self.net.finetune(x)
            loss = self.criterion(prob, y)
            loss.backward()
            self.optimizer.step()
            acc = torch.eq(torch.argmax(prob, dim=1), y).float().mean()
            pbar.set_description(f"loss: {loss.item()} acc: {acc.item()}")

    def train(self, data, label, iters):
        data = data.to(device)
        label = label.to(device)
        self.net.train()
        pbar = tqdm.trange(iters)
        for i in pbar:
            self.optimizer.zero_grad()
            prob = self.net.fewshot(data)
            loss = self.criterion(prob, label)
            loss.backward()
            self.optimizer.step()
            acc = torch.eq(torch.argmax(prob, dim=1), label).float().mean()
            pbar.set_description(f"loss: {loss.item()} acc: {acc.item()}")

    def eval(self, data, label):
        data = data.to(device)
        label = label.to(device)
        self.net.eval()
        prob = self.net.fewshot(data)
        pred = torch.argmax(prob, dim=1)
        acc = torch.eq(pred, label).float().mean()

        return acc.item()

    def run_pretrain(self):
        data = train_dict[b'images']
        data = data.reshape([data.shape[0], 1, 28, 28])
        data = data.astype(np.float32) / 255.
        print(np.unique(data))
        raise ValueError
        data = torch.from_numpy(data)
        label = train_dict[b'labels']
        label = label.astype(np.int64)
        label = torch.from_numpy(label)
        self.initialize()
        self.pretrain(data, label)

        return self.net.state_dict()

    def run_vanilla(self, state_dict):
        self.testset.initialize()
        metrics = []
        for i in range(self.testset.size):
            dataset = self.testset.next()
            dataset = to_classifier_format(dataset)
            self.initialize(state_dict)
            self.train(dataset['x'], dataset['y'], 10)
            acc = self.eval(dataset['q'], dataset['t'])
            metrics.append(acc)
            print(f'eval acc: {acc}')
        return np.mean(metrics)

    def run_augment(self, state_dict):
        self.testset.initialize()
        metrics = []
        for i in range(self.testset.size):
            dataset = self.testset.next()
            dataset = augment(dataset, args.G)
            dataset = to_classifier_format(dataset)
            self.initialize(state_dict)
            self.train(dataset['x'], dataset['y'], 100)
            acc = self.eval(dataset['q'], dataset['t'])
            metrics.append(acc)
            print(f'eval acc: {acc}')
        return np.mean(metrics)

save_dir = f'{params.exp_dir}/evaluate/fewshot_ft/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')
log_file.write(json.dumps(vars(args)))
log_file.write('\n\n')

runner = Runner()

state_dict = runner.run_pretrain()
torch.save(state_dict, f'{save_dir}/pretrain.pth')

state_dict = torch.load(f'{save_dir}/pretrain.pth')

acc = runner.run_vanilla(state_dict)
log_file.write(f'vanilla: {acc}\n')

acc = runner.run_augment(state_dict)
log_file.write(f'augment: {acc}\n')

log_file.close()