import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .base import BaseModel
from .fns_encoder import *
from .attention import Attention
from .networks import dense_nn

class ACIdp(BaseModel):
    def __init__(self, hps):
        self.prior_net = self.posterior_net = IdpLatentEncoder(hps)
        self.self_attn = Attention()
        self.generator_net = Generator(hps)
        super(ACIdp, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acidp', reuse=tf.AUTO_REUSE):
            self.xc = tf.placeholder(tf.float32, [None, self.hps.set_size, None, self.hps.x_dim])
            self.yc = tf.placeholder(tf.float32, [None, self.hps.set_size, None, self.hps.y_dim])
            self.xt = tf.placeholder(tf.float32, [None, self.hps.set_size, None, self.hps.x_dim])
            self.yt = tf.placeholder(tf.float32, [None, self.hps.set_size, None, self.hps.y_dim])

            # reshape
            B = tf.shape(self.xc)[0]
            K = self.hps.set_size
            Nc = tf.shape(self.xc)[2]
            Nt = tf.shape(self.xt)[2]
            dx = self.hps.x_dim
            dy = self.hps.y_dim

            xc = tf.reshape(self.xc, [B*K, Nc, dx])
            yc = tf.reshape(self.yc, [B*K, Nc, dy])
            xt = tf.reshape(self.xt, [B*K, Nt, dx])
            yt = tf.reshape(self.yt, [B*K, Nt, dy])

            # prior
            prior_inputs = tf.concat([xc, yc], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()

            # self attention embedding
            ctx_rep = tf.concat([xc, yc], axis=-1)
            ctx_rep = dense_nn(ctx_rep, self.hps.attn_value_hids, self.hps.attn_dim, name='attn_value')
            xt_rep = dense_nn(xt, self.hps.attn_key_hids[:-1], self.hps.attn_key_hids[-1], name='attn_key')
            xc_rep = dense_nn(xc, self.hps.attn_key_hids[:-1], self.hps.attn_key_hids[-1], name='attn_key')
            cs = self.self_attn(xt_rep, xc_rep, ctx_rep) #[B*K, Nt, d]

            # posterior
            posterior_inputs = tf.concat([xt, yt], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            kl = tfd.kl_divergence(posterior, prior)
            kl = tf.reduce_sum(kl, axis=1)
            tf.summary.scalar('kl', tf.reduce_mean(kl))

            # generator p(yt | xt, z)
            cz = tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,Nt,1])
            generator_inputs = tf.concat([xt, cz, cs], axis=-1)
            generator_dist = self.generator_net(generator_inputs)
            log_likel = generator_dist.log_prob(yt) # [B*K,Nt]
            log_likel = tf.reduce_mean(log_likel, axis=1)
            tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

            # elbo
            elbo = log_likel - kl / tf.cast(Nt, tf.float32) # [B*K]
            self.elbo = tf.reduce_mean(tf.reshape(elbo, [B,K]), axis=1)
            self.metric = self.elbo
            self.loss = tf.reduce_mean(-self.elbo)
            tf.summary.scalar('loss', self.loss)

            # sample
            cz = tf.tile(tf.expand_dims(prior_sample, axis=1), [1,Nt,1])
            sampler_inputs = tf.concat([xt, cz, cs], axis=-1)
            sampler_dist = self.generator_net(sampler_inputs)
            sample = sampler_dist.sample() #[B*K,Nt,dy]
            self.sample = tf.reshape(sample, [B,K,Nt,dy])
            mean = sampler_dist.mean()
            self.mean = tf.reshape(mean, [B,K,Nt,dy])
            std = sampler_dist.stddev()
            self.std = tf.reshape(std, [B,K,Nt,dy])

    def execute(self, cmd, batch):
        return self.sess.run(cmd, 
            {self.xc:batch['xc'], 
             self.yc:batch['yc'],
             self.xt:batch['xt'],
             self.yt:batch['yt']})