import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .base import BaseModel
from .acflow import ACFlow
from .im_encoder import Generator

class ACIdp(BaseModel):
    def __init__(self, hps):
        self.acflow = ACFlow(hps)
        super(ACIdp, self).__init__(hps)

    def build_net(self):
        self.x = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
        self.b = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
        self.m = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
        
        # generator p(x_u | x_o)
        x = tf.reshape(self.x, [-1]+self.hps.image_shape)
        b = tf.reshape(self.b, [-1]+self.hps.image_shape)
        m = tf.reshape(self.m, [-1]+self.hps.image_shape)
        log_likel = self.acflow.forward(x, b, m)
        log_likel = tf.reshape(log_likel, [-1,self.hps.set_size])
        self.set_metric = log_likel
        log_likel = tf.reduce_mean(log_likel, axis=1)
        tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

        # loss
        self.metric = log_likel
        self.loss = tf.reduce_mean(-log_likel)

        # sample
        x = tf.reshape(self.x, [-1]+self.hps.image_shape)
        b = tf.reshape(self.b, [-1]+self.hps.image_shape)
        m = tf.reshape(self.m, [-1]+self.hps.image_shape)
        sample = self.acflow.inverse(x, b, m)
        self.sample = tf.reshape(sample, [-1,self.hps.set_size]+self.hps.image_shape)

class ACIdpG(BaseModel):
    def __init__(self, hps):
        self.generator = Generator(hps)
        super(ACIdpG, self).__init__(hps)

    def build_net(self):
        self.x = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
        self.b = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
        self.m = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)

        # generator p(x_u | x_o)
        x = tf.reshape(self.x, [-1]+self.hps.image_shape)
        b = tf.reshape(self.b, [-1]+self.hps.image_shape)
        m = tf.reshape(self.m, [-1]+self.hps.image_shape)
        dist = self.generator(x, None, b, m)
        log_likel = dist.log_prob(x)
        log_likel = tf.reduce_sum(log_likel * m * (1-b), axis=(1,2,3))
        log_likel = tf.reshape(log_likel, [-1,self.hps.set_size])
        self.set_metric = log_likel
        log_likel = tf.reduce_mean(log_likel, axis=1)
        tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

        # loss
        self.metric = log_likel
        self.loss = tf.reduce_mean(-log_likel)

        # sample
        sample = dist.mean()
        self.sample = tf.reshape(sample, [-1,self.hps.set_size]+self.hps.image_shape)