import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from easydict import EasyDict as edict

from .base import BaseModel
from .ft_encoder import *
from .actan import Flow
from .flow.transforms import Transform

class ACSetBasic(BaseModel):
    def __init__(self, hps):
        self.prior_net = LatentEncoder(hps, name='prior')
        self.posterior_net = LatentEncoder(hps, name='posterior')
        self.peq_embed = SetXformer(hps)
        super(ACSetBasic, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acset_flow', reuse=tf.AUTO_REUSE):
            self.t = tf.placeholder(tf.float32, [None, None, self.hps.t_dim])
            self.x = tf.placeholder(tf.float32, [None, None, self.hps.f_dim])
            self.b = tf.placeholder(tf.float32, [None, None, self.hps.f_dim])
            self.m = tf.placeholder(tf.float32, [None, None, self.hps.f_dim])
            set_size = tf.shape(self.t)[1]

            # build transform
            self.acflow = Flow(edict(self.hps.acflow_params))

            # prior
            prior_inputs = tf.concat([self.t, self.x*self.b, self.b], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()

            # peq embedding
            peq_embed = self.peq_embed(prior_inputs)
            C = peq_embed.get_shape().as_list()[-1]
            cm = tf.reshape(peq_embed, [-1,C])

            # posterior
            posterior_inputs = tf.concat([self.t, self.x*self.m, self.m], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            kl = tfd.kl_divergence(posterior, prior)
            kl = tf.reduce_sum(kl, axis=1)
            tf.summary.scalar('kl', tf.reduce_mean(kl))

            # generator p(x_u | x_o, z)
            t = tf.reshape(self.t, [-1, self.hps.t_dim])
            x = tf.reshape(self.x, [-1, self.hps.f_dim])
            b = tf.reshape(self.b, [-1, self.hps.f_dim])
            m = tf.reshape(self.m, [-1, self.hps.f_dim])
            cv = tf.reshape(tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,set_size,1]), [-1,self.hps.latent_dim])
            c = tf.concat([cv, cm, t], axis=-1)
            log_likel = self.acflow.cond_forward(x, c, b, m)
            log_likel = tf.reshape(log_likel, [-1,set_size])
            log_likel = tf.reduce_sum(log_likel, axis=1)
            tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

            # elbo
            self.elbo = log_likel - kl
            self.metric = self.elbo
            self.loss = tf.reduce_mean(-self.elbo)
            tf.summary.scalar('loss', self.loss)

            # sample
            t = tf.reshape(self.t, [-1, self.hps.t_dim])
            x = tf.reshape(self.x, [-1, self.hps.f_dim])
            b = tf.reshape(self.b, [-1, self.hps.f_dim])
            m = tf.reshape(self.m, [-1, self.hps.f_dim])
            cv = tf.reshape(tf.tile(tf.expand_dims(prior_sample, axis=1), [1,set_size,1]), [-1,self.hps.latent_dim])
            c = tf.concat([cv, cm, t], axis=-1)
            sample = self.acflow.cond_inverse(x, c, b, m)
            self.sample = tf.reshape(sample, [-1, set_size, self.hps.f_dim])

    def execute(self, cmd, batch):
        return self.sess.run(cmd,
            {self.t:batch['t'],
             self.x:batch['x'],
             self.b:batch['b'],
             self.m:batch['m']})


class ACSetFlow(BaseModel):
    def __init__(self, hps):
        self.prior_net = LatentEncoder(hps, name='prior')
        self.posterior_net = LatentEncoder(hps, name='posterior')
        self.peq_embed = SetXformer(hps)
        super(ACSetFlow, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acset_flow', reuse=tf.AUTO_REUSE):
            self.t = tf.placeholder(tf.float32, [None, None, self.hps.t_dim])
            self.x = tf.placeholder(tf.float32, [None, None, self.hps.f_dim])
            self.b = tf.placeholder(tf.float32, [None, None, self.hps.f_dim])
            self.m = tf.placeholder(tf.float32, [None, None, self.hps.f_dim])
            set_size = tf.shape(self.t)[1]

            # build transform
            self.acflow = Flow(edict(self.hps.acflow_params))
            self.transform = Transform(edict(self.hps.trans_params))

            # prior
            prior_inputs = tf.concat([self.t, self.x*self.b, self.b], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()
            prior_sample, _ = self.transform.inverse(prior_sample)

            # peq embedding
            peq_embed = self.peq_embed(prior_inputs)
            C = peq_embed.get_shape().as_list()[-1]
            cm = tf.reshape(peq_embed, [-1,C])

            # posterior
            posterior_inputs = tf.concat([self.t, self.x*self.m, self.m], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            z_sample, logdet = self.transform.forward(posterior_sample)
            logp = tf.reduce_sum(prior.log_prob(z_sample), axis=1) + logdet
            kl = tf.reduce_sum(posterior.entropy(), axis=1) + logp
            tf.summary.scalar('kl', tf.reduce_mean(kl))

            # generator p(x_u | x_o, z)
            t = tf.reshape(self.t, [-1, self.hps.t_dim])
            x = tf.reshape(self.x, [-1, self.hps.f_dim])
            b = tf.reshape(self.b, [-1, self.hps.f_dim])
            m = tf.reshape(self.m, [-1, self.hps.f_dim])
            cv = tf.reshape(tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,set_size,1]), [-1,self.hps.latent_dim])
            c = tf.concat([cv, cm, t], axis=-1)
            log_likel = self.acflow.cond_forward(x, c, b, m)
            log_likel = tf.reshape(log_likel, [-1,set_size])
            log_likel = tf.reduce_sum(log_likel, axis=1)
            tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

            # elbo
            self.elbo = log_likel + kl
            self.metric = self.elbo

            # loss
            loss = log_likel + kl * self.hps.beta
            self.loss = tf.reduce_mean(-loss)
            tf.summary.scalar('loss', self.loss)

            # sample
            t = tf.reshape(self.t, [-1, self.hps.t_dim])
            x = tf.reshape(self.x, [-1, self.hps.f_dim])
            b = tf.reshape(self.b, [-1, self.hps.f_dim])
            m = tf.reshape(self.m, [-1, self.hps.f_dim])
            cv = tf.reshape(tf.tile(tf.expand_dims(prior_sample, axis=1), [1,set_size,1]), [-1,self.hps.latent_dim])
            c = tf.concat([cv, cm, t], axis=-1)
            sample = self.acflow.cond_inverse(x, c, b, m)
            self.sample = tf.reshape(sample, [-1, set_size, self.hps.f_dim])

    def execute(self, cmd, batch):
        return self.sess.run(cmd,
            {self.t:batch['t'],
             self.x:batch['x'],
             self.b:batch['b'],
             self.m:batch['m']})