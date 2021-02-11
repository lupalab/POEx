import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .base import BaseModel
from .acflow import ACFlow
from .im_encoder import *
from .flow.transforms import Transform

class ACSetFlow(BaseModel):
    def __init__(self, hps):
        self.prior_net = LatentEncoder(hps, name='prior')
        self.posterior_net = LatentEncoder(hps, name='posterior')
        if hps.use_peq_embed:
            self.img_peq_embed = DeepImgXformer(hps)
        if hps.use_vec_embed:
            self.vec_peq_embed = VecImgXformer(hps)
        self.acflow = ACFlow(hps)
        super(ACSetFlow, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acset_flow', reuse=tf.AUTO_REUSE):
            self.t = tf.placeholder(tf.float32, [None, self.hps.set_size, self.hps.t_dim])
            self.x = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.b = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.m = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)

            # build transform
            self.transform = Transform(self.hps)

            # tile
            H, W, C = self.hps.image_shape
            t = tf.expand_dims(tf.expand_dims(self.t, axis=2), axis=3)
            t = tf.tile(t, [1, 1, H, W, 1])

            # prior
            prior_inputs = tf.concat([t, self.x*self.b, self.b], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()
            prior_sample, _ = self.transform.inverse(prior_sample)

            # peq embedding
            cm = peq_embed = None
            if self.hps.use_peq_embed:
                peq_embed = self.img_peq_embed(prior_inputs)
                H, W, C = peq_embed.get_shape().as_list()[-3:]
                cm = tf.reshape(peq_embed, [-1,H,W,C])

            # vec embedding
            cp = None
            if self.hps.use_vec_embed:
                cp = self.vec_peq_embed(prior_inputs)
                C = cp.get_shape().as_list()[-1]
                cp = tf.reshape(cp, [-1, C])

            # posterior
            posterior_inputs = tf.concat([t, self.x*self.m, self.m], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            z_sample, logdet = self.transform.forward(posterior_sample)
            logp = tf.reduce_sum(prior.log_prob(z_sample), axis=1) + logdet
            kl = tf.reduce_sum(posterior.entropy(), axis=1) + logp
            tf.summary.scalar('kl', tf.reduce_mean(kl))

            # generator p(x_u | x_o, z, t)
            t = tf.reshape(self.t, [-1, self.hps.t_dim])
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            m = tf.reshape(self.m, [-1]+self.hps.image_shape)
            cv = tf.reshape(tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            cv = tf.concat([cv, t], axis=-1) if cp is None else tf.concat([cv, cp, t], axis=-1)
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
            t = tf.reshape(self.t, [-1, self.hps.t_dim])
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            m = tf.reshape(self.m, [-1]+self.hps.image_shape)
            cv = tf.reshape(tf.tile(tf.expand_dims(prior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            cv = tf.concat([cv, t], axis=-1) if cp is None else tf.concat([cv, cp, t], axis=-1)
            sample = self.acflow.inverse(x, b, m, cv=cv, cm=cm)
            self.sample = tf.reshape(sample, [-1,self.hps.set_size]+self.hps.image_shape)

            # mae loss
            if self.hps.lambda_mae > 0:
                mae = tf.reduce_sum(tf.abs(self.sample-self.x) * self.m * (1 - self.b), axis=[2,3,4])
                num = tf.reduce_sum(self.m * (1-self.b), axis=[2,3,4])
                num = tf.maximum(num, tf.ones_like(num))
                mae_loss = tf.reduce_mean(mae / num)
                self.loss = self.loss + mae_loss * self.hps.lambda_mae
                self.metric = -tf.reduce_mean(mae / num, axis=1)

    def execute(self, cmd, batch):
        return self.sess.run(cmd,
            {self.t:batch['t'],
             self.x:batch['x'],
             self.b:batch['b'],
             self.m:batch['m']})

class ACSetFlowRes(ACSetFlow):
    def build_net(self):
        with tf.variable_scope('acset_flow', reuse=tf.AUTO_REUSE):
            self.t = tf.placeholder(tf.float32, [None, self.hps.set_size, self.hps.t_dim])
            self.x = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.b = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.m = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)

            # build transform
            self.transform = Transform(self.hps)

            # tile
            H, W, C = self.hps.image_shape
            t = tf.expand_dims(tf.expand_dims(self.t, axis=2), axis=3)
            t = tf.tile(t, [1, 1, H, W, 1])

            # prior
            prior_inputs = tf.concat([t, self.x*self.b, self.b], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()
            prior_sample, _ = self.transform.inverse(prior_sample)

            # peq embedding
            cm = peq_embed = None
            if self.hps.use_peq_embed:
                peq_embed = self.img_peq_embed(prior_inputs)
                H, W, C = peq_embed.get_shape().as_list()[-3:]
                cm = tf.reshape(peq_embed, [-1,H,W,C])

            # vec embedding
            cp = None
            if self.hps.use_vec_embed:
                cp = self.vec_peq_embed(prior_inputs)
                C = cp.get_shape().as_list()[-1]
                cp = tf.reshape(cp, [-1, C])

            # residual
            avg = tf.reduce_sum(self.x * self.b, axis=1, keepdims=True)
            num = tf.reduce_sum(self.b, axis=1, keepdims=True)
            num = tf.maximum(num, tf.ones_like(num))
            avg = avg / num
            avg = self.x * self.b + avg * (1-self.b)
            common_miss = tf.reduce_prod(1-self.b, axis=1, keepdims=True) # 1: missing
            common_value = tf.reduce_sum(self.x * self.b, axis=[1,2,3,4], keepdims=True)
            common_num = tf.reduce_sum(self.b, axis=[1,2,3,4], keepdims=True)
            common_num = tf.maximum(common_num, tf.ones_like(common_num))
            common_value = common_value / common_num
            avg = avg * (1-common_miss) + common_value * common_miss
            avg = tf.reshape(avg, [-1]+self.hps.image_shape)
            res = tf.reshape(self.x, [-1]+self.hps.image_shape) - avg
            # res = (res + 255) / 2

            # posterior
            posterior_inputs = tf.concat([t, self.x*self.m, self.m], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            z_sample, logdet = self.transform.forward(posterior_sample)
            logp = tf.reduce_sum(prior.log_prob(z_sample), axis=1) + logdet
            kl = tf.reduce_sum(posterior.entropy(), axis=1) + logp
            tf.summary.scalar('kl', tf.reduce_mean(kl))

            # generator p(x_u | x_o, z, t)
            t = tf.reshape(self.t, [-1, self.hps.t_dim])
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            m = tf.reshape(self.m, [-1]+self.hps.image_shape)
            cv = tf.reshape(tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            cv = tf.concat([cv, t], axis=-1) if cp is None else tf.concat([cv, cp, t], axis=-1)
            cr = tf.concat([cm, avg], axis=-1)
            log_likel = self.acflow.forward(res, b, m, cv=cv, cm=cr)
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
            t = tf.reshape(self.t, [-1, self.hps.t_dim])
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            m = tf.reshape(self.m, [-1]+self.hps.image_shape)
            cv = tf.reshape(tf.tile(tf.expand_dims(prior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            cv = tf.concat([cv, t], axis=-1) if cp is None else tf.concat([cv, cp, t], axis=-1)
            cr = tf.concat([cm, avg], axis=-1)
            sample = self.acflow.inverse(res, b, m, cv=cv, cm=cr)
            sample = sample + avg
            # sample = sample * 2 - 255 + avg
            sample = tf.clip_by_value(sample, 0., 255.)
            self.sample = tf.reshape(sample, [-1,self.hps.set_size]+self.hps.image_shape)

            # mae loss
            if self.hps.lambda_mae > 0:
                mae = tf.reduce_sum(tf.abs(self.sample-self.x) * self.m * (1 - self.b), axis=[2,3,4])
                num = tf.reduce_sum(self.m * (1-self.b), axis=[2,3,4])
                num = tf.maximum(num, tf.ones_like(num))
                mae_loss = tf.reduce_mean(mae / num)
                self.loss = self.loss + mae_loss * self.hps.lambda_mae
                self.metric = -tf.reduce_mean(mae / num, axis=1)

class ACSetFlowG(BaseModel):
    def __init__(self, hps):
        self.prior_net = PeqLatentEncoder(hps, name='prior')
        self.posterior_net = PeqLatentEncoder(hps, name='posterior')
        if hps.use_peq_embed:
            self.img_peq_embed = DeepImgXformer(hps)
        if hps.use_vec_embed:
            self.vec_peq_embed = VecImgXformer(hps)
        self.generator = Generator(hps)
        super(ACSetFlowG, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acset_flow', reuse=tf.AUTO_REUSE):
            self.t = tf.placeholder(tf.float32, [None, self.hps.set_size, self.hps.t_dim])
            self.x = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.b = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)
            self.m = tf.placeholder(tf.float32, [None, self.hps.set_size] + self.hps.image_shape)

            # build transform
            self.transform = Transform(self.hps)

            # tile
            H, W, C = self.hps.image_shape
            t = tf.expand_dims(tf.expand_dims(self.t, axis=2), axis=3)
            t = tf.tile(t, [1, 1, H, W, 1])

            # prior
            prior_inputs = tf.concat([t, self.x*self.b, self.b], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()
            prior_sample, _ = self.transform.inverse(prior_sample)

            # peq embedding
            cm = peq_embed = None
            if self.hps.use_peq_embed:
                peq_embed = self.img_peq_embed(prior_inputs)
                H, W, C = peq_embed.get_shape().as_list()[-3:]
                cm = tf.reshape(peq_embed, [-1,H,W,C])

            # vec embedding
            cp = None
            if self.hps.use_vec_embed:
                cp = self.vec_peq_embed(prior_inputs)
                C = cp.get_shape().as_list()[-1]
                cp = tf.reshape(cp, [-1, C])
                cp = tf.expand_dims(tf.expand_dims(cp, axis=1), axis=2)
                cp = tf.tile(cp, [1,H,W,1])

            # posterior
            posterior_inputs = tf.concat([t, self.x*self.m, self.m], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            z_sample, logdet = self.transform.forward(posterior_sample)
            logp = tf.reduce_sum(prior.log_prob(z_sample), axis=1) + logdet
            kl = tf.reduce_sum(posterior.entropy(), axis=1) + logp
            tf.summary.scalar('kl', tf.reduce_mean(kl))

            # generator p(x_u | x_o, z, t)
            t = tf.reshape(t, [-1,H,W,self.hps.t_dim])
            x = tf.reshape(self.x, [-1]+self.hps.image_shape)
            b = tf.reshape(self.b, [-1]+self.hps.image_shape)
            m = tf.reshape(self.m, [-1]+self.hps.image_shape)
            cv = tf.reshape(tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            cv = tf.expand_dims(tf.expand_dims(cv, axis=1), axis=2)
            cv = tf.tile(cv, [1,H,W,1])
            c = [t, cv]
            if cm is not None:
                c += [cm]
            if cp is not None:
                c += [cp]
            c = tf.concat(c, axis=-1)
            dist = self.generator(x, c, b, m)
            log_likel = dist.log_prob(x)
            # log_likel = tf.reduce_sum(log_likel * m * (1-b), axis=(1,2,3))
            log_likel = tf.reduce_sum(log_likel, axis=(1,2,3))
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
            cv = tf.reshape(tf.tile(tf.expand_dims(prior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            cv = tf.expand_dims(tf.expand_dims(cv, axis=1), axis=2)
            cv = tf.tile(cv, [1,H,W,1])
            c = [t, cv]
            if cm is not None:
                c += [cm]
            if cp is not None:
                c += [cp]
            c = tf.concat(c, axis=-1)
            dist = self.generator(x, c, b, m)
            sample = dist.mean()
            self.sample = tf.reshape(sample, [-1,self.hps.set_size]+self.hps.image_shape)

    def execute(self, cmd, batch):
        return self.sess.run(cmd,
            {self.t:batch['t'],
             self.x:batch['x'],
             self.b:batch['b'],
             self.m:batch['m']})

class ACSetFlowResG(ACSetFlowG):
    def __init__(self, hps):
        self.prior_net = PeqLatentEncoder(hps, name='prior')
        self.posterior_net = PeqLatentEncoder(hps, name='posterior')
        if hps.use_peq_embed:
            self.img_peq_embed = DeepImgXformer(hps)
        if hps.use_vec_embed:
            self.vec_peq_embed = VecImgXformer(hps)
        self.generator = ResGenerator(hps)
        super(ACSetFlowG, self).__init__(hps)
        