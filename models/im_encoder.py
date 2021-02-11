import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .networks import convnet, peq_convnet, peq_resblock
from .set_transformer import set_transformer, set_pooling
from .attention import Attention

class LatentEncoder(object):
    def __init__(self, hps, name='latent'):
        self.hps = hps
        self.name = name

    def __call__(self, x):
        '''
        x: [B,N,H,W,C]
        '''
        B,N,H,W,C = tf.shape(x)[0], tf.shape(x)[1], *x.get_shape().as_list()[2:]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, [-1,H,W,C])
            x = convnet(x, self.hps.latent_encoder_hidden, self.hps.latent_dim*2)
            x = tf.reshape(x, [B,N,self.hps.latent_dim*2])
            x = tf.reduce_mean(x, axis=1)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dense(x, self.hps.latent_dim*2, name='p')
            m, s = x[...,:self.hps.latent_dim], tf.nn.softplus(x[...,self.hps.latent_dim:])
            dist = tfd.Normal(loc=m, scale=s)

        return dist

class PeqLatentEncoder(object):
    def __init__(self, hps, name='peq_latent'):
        self.hps = hps
        self.name = name

        self.attention = Attention()

    def __call__(self, x):
        '''
        x: [B,N,H,W,C]
        '''
        B,N,H,W,C = tf.shape(x)[0], tf.shape(x)[1], *x.get_shape().as_list()[2:]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # peq embedding
            x = peq_convnet(x, self.hps.latent_encoder_hidden, self.hps.latent_dim*2, self.attention)
            # set transformer
            x = set_transformer(x, self.hps.set_xformer_hidden+[self.hps.latent_dim*2], 'set_xformer')
            x = set_pooling(x, 'set_pool')
            # distribution
            m, s = x[...,:self.hps.latent_dim], tf.nn.softplus(x[...,self.hps.latent_dim:])
            dist = tfd.Normal(loc=m, scale=s)

        return dist

class PeqXformerEncoder(object):
    def __init__(self, hps, name='peq_latent'):
        self.hps = hps
        self.name = name

        self.attention = Attention()

    def __call__(self, x):
        '''
        x: [B,N,H,W,C]
        '''
        B,N,H,W,C = tf.shape(x)[0], tf.shape(x)[1], *x.get_shape().as_list()[2:]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.hps.n_comps > 1:
                dim = self.hps.latent_dim * self.hps.n_comps * 2 + self.hps.n_comps
                # peq embedding
                x = peq_convnet(x, self.hps.latent_encoder_hidden, dim, self.attention)
                # set transformer
                x = set_transformer(x, self.hps.set_xformer_hidden, 'set_xformer')
                x = tf.layers.dense(x, dim)
                # distribution
                logits = x[...,:self.hps.n_comps]
                mean = x[...,self.hps.n_comps:-self.hps.latent_dim * self.hps.n_comps]
                mean = tf.reshape(mean, [B,N,self.hps.n_comps,self.hps.latent_dim])
                sigma = tf.nn.softplus(x[...,-self.hps.latent_dim * self.hps.n_comps:])
                sigma = tf.reshape(sigma, [B,N,self.hps.n_comps,self.hps.latent_dim])
                dist = tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(logits=logits),
                    components_distribution=tfd.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
                )
            else:
                dim = self.hps.latent_dim * 2
                # peq embedding
                x = peq_convnet(x, self.hps.latent_encoder_hidden, dim, self.attention)
                # set transformer
                x = set_transformer(x, self.hps.set_xformer_hidden, 'set_xformer')
                x = tf.layers.dense(x, dim)
                # distribution
                m, s = x[...,:self.hps.latent_dim], tf.nn.softplus(x[...,self.hps.latent_dim:])
                dist = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)

        return dist

class ImgEncoder(object):
    def __init__(self, hps, name='img_encoder'):
        self.hps = hps
        self.name = name

    def __call__(self, x):
        '''
        independent embedding
        x: [B,N,H,W,C]
        '''
        B,N,H,W,C = tf.shape(x)[0], tf.shape(x)[1], *x.get_shape().as_list()[2:]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # downsample
            x = tf.reshape(x, [-1,H,W,C])
            for d in self.hps.img_encoder_hids:
                x = tf.layers.conv2d(x, d, 3, strides=(1,1), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.conv2d(x, d, 3, strides=(2,2), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                H, W, C = H//2, W//2, d
            # upsample
            for d in reversed(self.hps.img_encoder_hids):
                x = tf.layers.conv2d_transpose(x, d, 3, strides=(2,2), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.conv2d(x, d, 3, strides=(1,1), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                H, W, C = H*2, W*2, d
            # reshape
            x = tf.reshape(x, [B,N,H,W,C])
            
        return x


class DeepImgXformer(object):
    def __init__(self, hps, name='deep_img_xformer'):
        self.hps = hps
        self.name = name

    def __call__(self, x):
        '''
        permutation equivariant embedding
        x: [B,N,H,W,C]
        '''
        B,N,H,W,C = tf.shape(x)[0], tf.shape(x)[1], *x.get_shape().as_list()[2:]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # downsample
            x = tf.reshape(x, [-1,H,W,C])
            for d in self.hps.img_xformer_hids:
                x = tf.layers.conv2d(x, d, 3, strides=(1,1), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.conv2d(x, d, 3, strides=(2,2), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                H, W, C = H//2, W//2, d
            x = tf.reshape(x, [B,N,H,W,C])
            # res blocks
            for i in range(self.hps.img_xformer_res):
                x = peq_resblock(x, d, Attention(name=f'res_{i}'), name=f'res_{i}')
            # upsample
            x = tf.reshape(x, [-1,H,W,C])
            for d in reversed(self.hps.img_xformer_hids):
                x = tf.layers.conv2d_transpose(x, d, 3, strides=(2,2), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.conv2d(x, d, 3, strides=(1,1), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                H, W, C = H*2, W*2, d
            # reshape
            x = tf.reshape(x, [B,N,H,W,C])
            
        return x

class ImgXformer(object):
    def __init__(self, hps, name='img_xformer'):
        self.hps = hps
        self.name = name

        self.attention = Attention()

    def __call__(self, x):
        '''
        permutation equivariant embedding
        x: [B,N,H,W,C]
        '''
        B,N,H,W,C = tf.shape(x)[0], tf.shape(x)[1], *x.get_shape().as_list()[2:]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # downsample
            x = tf.reshape(x, [-1,H,W,C])
            for d in self.hps.img_xformer_hids:
                x = tf.layers.conv2d(x, d, 3, strides=(1,1), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.conv2d(x, d, 3, strides=(2,2), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                H, W, C = H//2, W//2, d
            x = tf.reshape(x, [B,N,H,W,C])
            # attention across set dimension
            x = tf.reshape(tf.transpose(x, [0,2,3,1,4]), [B*H*W,N,C])
            rep = self.attention(x, x, x)
            x += rep
            x = tf.transpose(tf.reshape(x, [B,H,W,N,C]), [0,3,1,2,4])
            # upsample
            x = tf.reshape(x, [-1,H,W,C])
            for d in reversed(self.hps.img_xformer_hids):
                x = tf.layers.conv2d_transpose(x, d, 3, strides=(2,2), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.conv2d(x, d, 3, strides=(1,1), padding='same')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                H, W, C = H*2, W*2, d
            # reshape
            x = tf.reshape(x, [B,N,H,W,C])
            
        return x

class VecImgXformer(object):
    def __init__(self, hps, name='vec_xformer'):
        self.hps = hps
        self.name = name

    def __call__(self, x):
        '''
        permutation equivariant embedding
        x: [B,N,H,W,C]
        '''
        B,N,H,W,C = tf.shape(x)[0], tf.shape(x)[1], *x.get_shape().as_list()[2:]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = peq_convnet(x, self.hps.vec_xformer_hids,  self.hps.vec_xformer_hids[-1], Attention())
            # set transformer
            x = set_transformer(x, self.hps.set_xformer_hidden, 'set_xformer')

        return x

class Generator(object):
    def __init__(self, hps, name='generator'):
        self.hps = hps
        self.name = name

    def conv_block(self, x, dim, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(x, dim, 3, strides=(1,1), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, dim, 3, strides=(1,1), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)

        return x

    def deconv_block(self, x, dim, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d_transpose(x, dim, 3, strides=(2,2), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, dim, 3, strides=(1,1), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)

        return x

    def consistency(self, x, p, b):
        p = x * b + p * (1-b)

        return p

    def __call__(self, x, c, b, m):
        '''
        shape: [B,H,W,C]
        p(x_u | x_o, c)
        '''
        C = x.get_shape().as_list()[-1]
        if c is not None:
            h = tf.concat([x*b, b, c], axis=-1)
        else:
            h = tf.concat([x*b, b], axis=-1)
        dim = self.hps.generator_dim
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h = self.conv_block(h, dim, 'down0')
            stack = []
            for i in range(self.hps.generator_layers):
                stack.append(h)
                h = tf.layers.average_pooling2d(h, 2, 2)
                h = self.conv_block(h, dim*2, f'down{i+1}')
                dim = dim * 2

            for i in range(self.hps.generator_layers):
                h = self.deconv_block(h, dim//2, f'up{i}')
                h = tf.concat([h, stack.pop()], axis=-1)
                dim = dim // 2
            h = self.conv_block(h, dim, f'up{i+1}')

            h = tf.layers.conv2d(h, dim//2, 1, name='out1')
            h = tf.nn.leaky_relu(h)
            h = tf.layers.conv2d(h, C, 1, name='out2')
            h = tf.nn.leaky_relu(h)
            mean = tf.layers.conv2d(h, C, 1, name='mean')
            mean = self.consistency(x, mean, b)
            sigma = tf.ones_like(mean) * 0.1
            dist = tfd.Normal(loc=mean, scale=sigma)

            return dist

class ResGenerator(object):
    def __init__(self, hps, name='res_generator'):
        self.hps = hps
        self.name = name

    def conv_block(self, x, dim, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(x, dim, 3, strides=(1,1), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, dim, 3, strides=(1,1), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)

        return x

    def deconv_block(self, x, dim, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d_transpose(x, dim, 3, strides=(2,2), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, dim, 3, strides=(1,1), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)

        return x

    def consistency(self, x, p, b):
        p = x * b + p * (1-b)

        return p

    def __call__(self, x, c, b, m):
        '''
        shape: [B,H,W,C]
        p(x_u | x_o, c)
        '''
        B = tf.shape(x)[0]
        H,W,C = x.get_shape().as_list()[-3:]
        if c is not None:
            h = tf.concat([x*b, b, c], axis=-1)
        else:
            h = tf.concat([x*b, b], axis=-1)
        bs = tf.reshape(b, [B//self.hps.set_size, self.hps.set_size, H, W, C])
        xs = tf.reshape(x, [B//self.hps.set_size, self.hps.set_size, H, W, C])
        avg = tf.reduce_sum(xs * bs, axis=1, keepdims=True)
        num = tf.reduce_sum(bs, axis=1, keepdims=True)
        num = tf.maximum(num, tf.ones_like(num))
        avg = avg / num
        avg = xs * bs + avg * (1-bs)
        common_miss = tf.reduce_prod(1-bs, axis=1, keepdims=True) # 1: missing
        common_value = tf.reduce_sum(xs * bs, axis=[1,2,3,4], keepdims=True)
        common_num = tf.reduce_sum(bs, axis=[1,2,3,4], keepdims=True)
        common_num = tf.maximum(common_num, tf.ones_like(common_num))
        common_value = common_value / common_num
        avg = avg * (1-common_miss) + common_value * common_miss
        avg = tf.reshape(avg, [B,H,W,C])
        h = tf.concat([h, avg], axis=-1)
        dim = self.hps.generator_dim
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h = self.conv_block(h, dim, 'down0')
            stack = []
            for i in range(self.hps.generator_layers):
                stack.append(h)
                h = tf.layers.average_pooling2d(h, 2, 2)
                h = self.conv_block(h, dim*2, f'down{i+1}')
                dim = dim * 2

            for i in range(self.hps.generator_layers):
                h = self.deconv_block(h, dim//2, f'up{i}')
                h = tf.concat([h, stack.pop()], axis=-1)
                dim = dim // 2
            h = self.conv_block(h, dim, f'up{i+1}')

            h = tf.layers.conv2d(h, dim//2, 1, name='out1')
            h = tf.nn.leaky_relu(h)
            h = tf.layers.conv2d(h, C, 1, name='out2')
            h = tf.nn.leaky_relu(h)
            res = tf.layers.conv2d(h, C, 1, name='res')
            mean = res + avg
            # mean = self.consistency(x, mean, b)
            sigma = tf.ones_like(mean) * 0.1
            dist = tfd.Normal(loc=mean, scale=sigma)

            return dist