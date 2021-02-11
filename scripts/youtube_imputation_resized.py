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
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils.hparams import HParams
from models import get_model

from models.img_metrics.ssim import ssim_np

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--quality', type=str, default='480p')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

_SIZE = {
    'small': [64,64],
    '240p': [420,240],
    '480p': [840,480],
    '720p': [1280,720],
}[args.quality]

# data
data_path = '/data/youtube-vos/test/JPEGImages/'
videos = [f'{data_path}/{d}' for d in os.listdir(data_path)]
video_list = []
for v in videos:
    if os.path.isdir(v):
        images = []
        for m in os.listdir(v):
            if m.endswith('.jpg') and not m.startswith('.'):
                images.append(f'{v}/{m}')
        video_list.append(images)

def position_embedding(position, embed_dim):
    div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
    e0 = np.sin(position * div_term)
    e1 = np.cos(position * div_term)
    e = np.stack([e0, e1], axis=-1).reshape([-1,embed_dim])

    return e.astype(np.float32)

def get_time(index, total_frames, embed_dim):
    index = np.array(index, dtype=np.float32)
    time = (index * 1.0 / total_frames).astype(np.float32)
    time = np.expand_dims(time, axis=-1)

    if params.t_dim > 1:
        return position_embedding(time, embed_dim)
    return time

def read(files):
    x = []
    x_org = []
    for fname in files:
        img = Image.open(fname)
        x_org.append(np.array(img.resize(_SIZE)))
        x.append(np.array(img.resize((64,64))))
    x = np.stack(x, axis=0)
    x_org = np.stack(x_org, axis=0)

    return x, x_org

def generate_mask(x):
    mask = np.ones_like(x)
    mask[:,24:40,24:40] = 0
    return mask

class VideoDataset(Dataset):
    def __init__(self):
        self.video_list = video_list

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        rng = np.random.RandomState(index)
        video = self.video_list[index]
        replace = len(video) < params.set_size
        inds = rng.choice(len(video), params.set_size, replace=replace)
        t = get_time(inds, len(video), params.t_dim)
        files = [video[i] for i in inds]
        x, x_org = read(files)
        b = generate_mask(x)
        m = np.ones_like(b)

        return t, x, b, m, x_org

dataset = VideoDataset()
dataloader = DataLoader(dataset, batch_size=params.batch_size)

# model
model = get_model(params)
model.load()

# run
save_dir = f'{params.exp_dir}/evaluate/video_impute/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')
log_file.write(f'### {args.quality} ###\n')

def resize(x, size, rsample):
    x = x.astype(np.uint8)
    B,N,H,W,C = x.shape
    img = []
    for i in range(B):
        for j in range(N):
            im = Image.fromarray(x[i,j]).resize(size, rsample)
            img.append(np.array(im))
    img = np.stack(img, axis=0)
    img = np.reshape(img, [B,N,size[1],size[0],3])

    return img

def batch_ssim(x, s):
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
        assert np.max(sample) == 255 and np.max(batch['x_org']) == 255
        x_org = batch['x_org']
        b = resize(batch['b'], _SIZE, rsample=Image.NEAREST)
        s = resize(sample, _SIZE, rsample=Image.BICUBIC)
        s = x_org * b + s * (1-b)
        mse = np.sum(np.square(s-x_org), axis=(2,3,4))
        num = np.sum(1-b, axis=(2,3,4))
        mse = np.mean(mse/num, axis=1)
        average_mse.append(mse)
        average_ssim.append(batch_ssim(x_org, s))
    average_mse = np.mean(np.stack(average_mse, axis=0), axis=0)
    average_ssim = np.mean(np.stack(average_ssim, axis=0), axis=0)

    return average_mse, average_ssim

def visualize(batch, prefix):
    sample = model.execute(model.sample, batch)
    x = batch['x']
    x_org = batch['x_org']
    b = resize(batch['b'], _SIZE, rsample=Image.NEAREST)
    s = resize(sample, _SIZE, rsample=Image.BICUBIC)
    s = x_org * b + s * (1-b)
    B,N,H,W,C = s.shape
    for i in range(B):
        ss, xx, bb = s[i], x_org[i], b[i]
        ss = np.transpose(ss, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        xx = np.transpose(xx, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        bb = np.transpose(bb, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        xo = xx * bb + (1-bb) * 128
        img = np.concatenate([xx, xo, ss]).astype(np.uint8)
        plt.imsave(f'{prefix}_{i}_org.png', img)

        ss, xx, bb = sample[i], x[i], batch['b'][i]
        ss = np.transpose(ss, [1,0,2,3]).reshape(64,64*N,C).squeeze()
        xx = np.transpose(xx, [1,0,2,3]).reshape(64,64*N,C).squeeze()
        bb = np.transpose(bb, [1,0,2,3]).reshape(64,64*N,C).squeeze()
        xo = xx * bb + (1-bb) * 128
        img = np.concatenate([xx, xo, ss]).astype(np.uint8)
        plt.imsave(f'{prefix}_{i}.png', img)

# test
save_path = f'{save_dir}/test/'
os.makedirs(save_path, exist_ok=True)

test_mse = []
test_ssim = []
for t,x,b,m,x_org in dataloader:
    batch = {
        't': t.numpy(),
        'x': x.numpy(),
        'b': b.numpy(),
        'm': m.numpy(),
        'x_org': x_org.numpy()
    }
    res = evaluate(batch)
    test_mse.append(res[0])
    test_ssim.append(res[1])
test_mse = np.mean(np.concatenate(test_mse))
test_psnr = 20 * np.log10(255.) - 10 * np.log10(test_mse)
test_ssim = np.mean(np.concatenate(test_ssim))
log_file.write(f'test_mse: {test_mse}\n')
log_file.write(f'test_psnr: {test_psnr}\n')
log_file.write(f'test_ssim: {test_ssim}\n')

for i, (t,x,b,m,x_org) in enumerate(dataloader):
    if i >= 10: break
    prefix = f'{save_path}/batch_{i}'
    batch = {
        't': t.numpy(),
        'x': x.numpy(),
        'b': b.numpy(),
        'm': m.numpy(),
        'x_org': x_org.numpy()
    }
    visualize(batch, prefix)

log_file.close()