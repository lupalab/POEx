import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .networks import dense_nn
from .set_transformer import set_transformer, set_pooling

class IdpLatentEncoder(object):
    def __init__(self, hps, name='latent'):
        self.hps = hps
        self.name = name

    def __call__(self, x):
        '''
        x: [B,N,d]
        '''
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = dense_nn(x, self.hps.latent_encoder_hidden, self.hps.latent_dim*2, norm=False)
            x = tf.reduce_mean(x, axis=1)
            x = tf.layers.dense(x, self.hps.latent_dim*2, name='d')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dense(x, self.hps.latent_dim*2, name='p')
            m, s = x[...,:self.hps.latent_dim], tf.sigmoid(x[...,self.hps.latent_dim:])
            s = 0.1 + 0.9 * s
            dist = tfd.Normal(loc=m, scale=s)

        return dist

class LatentEncoder(object):
    def __init__(self, hps, name='latent'):
        self.hps = hps
        self.name = name

    def __call__(self, x):
        '''
        x: [B,K,N,d]
        '''
        B = tf.shape(x)[0]
        N = tf.shape(x)[2]
        _,K,_,d = x.get_shape().as_list()
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, [B*K,N,d])
            x = set_transformer(x, self.hps.latent_encoder_hidden, name='set_xformer')
            x = set_pooling(x, name='set_pool')
            x = tf.reshape(x, [B,K,self.hps.latent_encoder_hidden[-1]])
            x = tf.layers.dense(x, self.hps.latent_dim*2, name='d')
            x = tf.reduce_mean(x, axis=1)
            x = tf.layers.dense(x, self.hps.latent_dim*2, name='l')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dense(x, self.hps.latent_dim*2, name='p')
            m, s = x[...,:self.hps.latent_dim], tf.sigmoid(x[...,self.hps.latent_dim:])
            s = 0.1 + 0.9 * s
            dist = tfd.Normal(loc=m, scale=s)

        return dist

class Generator(object):
    def __init__(self, hps, name='generator'):
        self.hps = hps
        self.name = name

    def __call__(self, x):
        '''
        x: [...,d]
        '''
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = dense_nn(x, self.hps.generator_hidden, self.hps.y_dim*2)
            m, s = x[...,:self.hps.y_dim], tf.nn.softplus(x[...,self.hps.y_dim:])
            s = 0.1 + 0.9 * s
            dist = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)

        return dist


