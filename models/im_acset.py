import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .base import BaseModel
from .acflow import ACFlow
from .im_encoder import *
from .flow.transforms import Transform
from .flowscan.transforms import SetTransform

class ACSetBasic(BaseModel):
    def __init__(self, hps):
        self.latent = LatentEncoder(hps)
        if hps.use_peq_embed:
            self.peq_embed = ImgXformer(hps)
        self.acflow = ACFlow(hps)
        super(ACSetBasic, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acset_basic', reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.b = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            # prior
            prior_inputs = tf.concat([self.x*self.b, self.b], axis=-1)
            prior = self.latent(prior_inputs)
            prior_sample = prior.sample()

            # peq embedding
            cm = peq_embed = None
            if self.hps.use_peq_embed:
                peq_embed = self.peq_embed(prior_inputs)
                H, W, C = peq_embed.get_shape().as_list()[-3:]
                cm = tf.reshape(peq_embed, [-1,H,W,C])

            # posterior
            o = tf.ones_like(self.b)
            posterior_inputs = tf.concat([self.x, o], axis=-1)
            posterior = self.latent(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            kl = tfd.kl_divergence(posterior, prior)
            kl = tf.reduce_sum(kl, axis=1)
            tf.summary.scalar('kl', tf.reduce_mean(kl))

            # generator p(x_u | x_o, z)
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            cv = tf.reshape(tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            log_likel = self.acflow.forward(x, b, cv=cv, cm=cm)
            log_likel = tf.reshape(log_likel, [-1,self.hps.set_size])
            log_likel = tf.reduce_mean(log_likel, axis=1)
            tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

            # elbo
            self.elbo = log_likel - kl / self.hps.set_size
            self.metric = self.elbo
            self.loss = tf.reduce_mean(-self.elbo)
            tf.summary.scalar('loss', self.loss)

            # sample
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            cv = tf.reshape(tf.tile(tf.expand_dims(prior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            sample = self.acflow.inverse(x, b, cv=cv, cm=cm)
            self.sample = tf.reshape(sample, [-1,self.hps.set_size]+self.hps.image_shape)

    
class ACSetFlow(BaseModel):
    def __init__(self, hps):
        self.prior_net = LatentEncoder(hps, name='prior')
        self.posterior_net = LatentEncoder(hps, name='posterior')
        if hps.use_peq_embed:
            self.peq_embed = ImgXformer(hps)
        self.acflow = ACFlow(hps)
        super(ACSetFlow, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acset_flow', reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.b = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.m = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            
            # build transform
            self.transform = Transform(self.hps)

            # prior
            prior_inputs = tf.concat([self.x*self.b, self.b], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()
            prior_sample, _ = self.transform.inverse(prior_sample)

            # peq embedding
            cm = peq_embed = None
            if self.hps.use_peq_embed:
                peq_embed = self.peq_embed(prior_inputs)
                H, W, C = peq_embed.get_shape().as_list()[-3:]
                cm = tf.reshape(peq_embed, [-1,H,W,C])

            # posterior
            posterior_inputs = tf.concat([self.x*self.m, self.m], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            z_sample, logdet = self.transform.forward(posterior_sample)
            logp = tf.reduce_sum(prior.log_prob(z_sample), axis=1) + logdet
            kl = tf.reduce_sum(posterior.entropy(), axis=1) + logp
            tf.summary.scalar('kl', tf.reduce_mean(kl))

            # generator p(x_u | x_o, z)
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            m = tf.reshape(self.m, [-1]+self.hps.image_shape)
            cv = tf.reshape(tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            log_likel = self.acflow.forward(x, b, m, cv=cv, cm=cm)
            log_likel = tf.reshape(log_likel, [-1,self.hps.set_size])
            self.set_metric = self.set_elbo = log_likel + tf.expand_dims(kl, axis=1) / self.hps.set_size
            log_likel = tf.reduce_mean(log_likel, axis=1)
            tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

            # elbo
            self.metric = self.elbo = log_likel + kl / self.hps.set_size
            loss = log_likel + kl / self.hps.set_size * self.hps.beta
            self.loss = tf.reduce_mean(-loss)
            tf.summary.scalar('loss', self.loss)

            # sample
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            m = tf.reshape(self.m, [-1]+self.hps.image_shape)
            cv = tf.reshape(tf.tile(tf.expand_dims(prior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            sample = self.acflow.inverse(x, b, m, cv=cv, cm=cm)
            self.sample = tf.reshape(sample, [-1,self.hps.set_size]+self.hps.image_shape)


class ACSetFlowScan(BaseModel):
    def __init__(self, hps):
        self.prior_net = PeqLatentEncoder(hps, name='prior')
        self.posterior_net = PeqLatentEncoder(hps, name='posterior')
        if hps.use_peq_embed:
            self.peq_embed = ImgXformer(hps)
        self.acflow = ACFlow(hps)
        super(ACSetFlowScan, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acset_flowscan', reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.b = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)

            # build transform
            self.transform = SetTransform(self.hps)

            # prior
            prior_inputs = tf.concat([self.x*self.b, self.b], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()
            prior_sample, _ = self.transform.inverse(prior_sample)

            # peq embedding
            cm = peq_embed = None
            if self.hps.use_peq_embed:
                peq_embed = self.peq_embed(prior_inputs)
                H, W, C = peq_embed.get_shape().as_list()[-3:]
                cm = tf.reshape(peq_embed, [-1,H,W,C])

            # posterior
            o = tf.ones_like(self.b)
            posterior_inputs = tf.concat([self.x, o], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            z_sample, logdet = self.transform.forward(posterior_sample)
            logp = tf.reduce_sum(prior.log_prob(z_sample), axis=[1,2]) + logdet
            kl = tf.reduce_sum(posterior.entropy(), axis=[1,2]) + logp

            # generator p(x_u | x_o, z)
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            cv = tf.reshape(posterior_sample, [-1, self.hps.latent_dim])
            log_likel = self.acflow.forward(x, b, cv=cv, cm=cm)
            log_likel = tf.reshape(log_likel, [-1,self.hps.set_size])
            log_likel = tf.reduce_mean(log_likel, axis=1)
            tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

            # elbo
            self.elbo = log_likel + kl / self.hps.set_size
            self.metric = self.elbo
            self.loss = tf.reduce_mean(-self.elbo)
            tf.summary.scalar('loss', self.loss)

            # sample
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            cv = tf.reshape(prior_sample, [-1,self.hps.latent_dim])
            sample = self.acflow.inverse(x, b, cv=cv, cm=cm)
            self.sample = tf.reshape(sample, [-1,self.hps.set_size]+self.hps.image_shape)


class ACSetXformer(BaseModel):
    def __init__(self, hps):
        self.prior_net = PeqXformerEncoder(hps, name='prior')
        self.posterior_net = PeqLatentEncoder(hps, name='posterior')
        if hps.use_peq_embed:
            self.peq_embed = ImgXformer(hps)
        self.acflow = ACFlow(hps)
        super(ACSetXformer, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acset_xformer', reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.b = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)

            # prior
            prior_inputs = tf.concat([self.x*self.b, self.b], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()
            
            # peq embedding
            cm = peq_embed = None
            if self.hps.use_peq_embed:
                peq_embed = self.peq_embed(prior_inputs)
                H, W, C = peq_embed.get_shape().as_list()[-3:]
                cm = tf.reshape(peq_embed, [-1,H,W,C])

            # posterior
            o = tf.ones_like(self.b)
            posterior_inputs = tf.concat([self.x, o], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            logp = tf.reduce_sum(prior.log_prob(posterior_sample), axis=1)
            kl = tf.reduce_sum(posterior.entropy(), axis=[1,2]) + logp

            # generator p(x_u | x_o, z)
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            cv = tf.reshape(posterior_sample, [-1, self.hps.latent_dim])
            log_likel = self.acflow.forward(x, b, cv=cv, cm=cm)
            log_likel = tf.reshape(log_likel, [-1,self.hps.set_size])
            log_likel = tf.reduce_mean(log_likel, axis=1)
            tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

            # elbo
            self.elbo = log_likel + kl / self.hps.set_size
            self.metric = self.elbo
            self.loss = tf.reduce_mean(-self.elbo)
            tf.summary.scalar('loss', self.loss)

            # sample
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            cv = tf.reshape(prior_sample, [-1,self.hps.latent_dim])
            sample = self.acflow.inverse(x, b, cv=cv, cm=cm)
            self.sample = tf.reshape(sample, [-1,self.hps.set_size]+self.hps.image_shape)

