import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from easydict import EasyDict as edict

from .base import BaseModel
from .actan import Flow
from .pc_encoder import *
from .flow.transforms import Transform
from .tf_pc_distance.pc_distance import chamfer
from .tf_pc_distance.pc_kde import kde

class ACSetFlow(BaseModel):
    def __init__(self, hps):
        self.prior_net = LatentEncoder(hps, name='prior')
        self.posterior_net = LatentEncoder(hps, name='posterior')
        if hps.use_peq_embed:
            self.peq_embed = SetXformer(hps)
        super(ACSetFlow, self).__init__(hps)
    
    def build_net(self):
        with tf.variable_scope('acset_flow', reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32, [None, self.hps.set_size, self.hps.dimension])
            self.b = tf.placeholder(tf.float32, [None, self.hps.set_size, self.hps.dimension])
            self.m = tf.placeholder(tf.float32, [None, self.hps.set_size, self.hps.dimension])

            # build transform
            self.acflow = Flow(edict(self.hps.acflow_params))
            self.transform = Transform(edict(self.hps.trans_params))

            # prior
            prior_inputs = tf.concat([self.x*self.b, self.b], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample()
            prior_sample, _ = self.transform.inverse(prior_sample)

            # peq embedding
            cm = peq_embed = None
            if self.hps.use_peq_embed:
                peq_embed = self.peq_embed(prior_inputs)
                C = peq_embed.get_shape().as_list()[-1]
                cm = tf.reshape(peq_embed, [-1,C])
            
            # posterior
            posterior_inputs = tf.concat([self.x*self.m, self.m], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample()

            # kl term
            z_sample, logdet = self.transform.forward(posterior_sample)
            logp = tf.reduce_sum(prior.log_prob(z_sample), axis=1) + logdet
            kl = tf.reduce_sum(posterior.entropy(), axis=1) + logp

            # generator p(x_u | x_o, z)
            x = tf.reshape(self.x, [-1,self.hps.dimension])
            b = tf.reshape(self.b, [-1,self.hps.dimension])
            m = tf.reshape(self.m, [-1,self.hps.dimension])
            cv = tf.reshape(tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            c = tf.concat([cv, cm], axis=-1)
            log_likel = self.acflow.cond_forward(x, c, b, m)
            log_likel = tf.reshape(log_likel, [-1,self.hps.set_size])
            self.set_metric = self.set_elbo = log_likel + tf.expand_dims(kl, axis=1) / self.hps.set_size
            log_likel = tf.reduce_mean(log_likel, axis=1)
            tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

            # elbo
            self.elbo = log_likel + kl / self.hps.set_size
            self.metric = self.elbo
            self.loss = tf.reduce_mean(-self.elbo)
            tf.summary.scalar('loss', self.loss)

            # sample
            x = tf.reshape(self.x, [-1,self.hps.dimension])
            b = tf.reshape(self.b, [-1,self.hps.dimension])
            m = tf.reshape(self.m, [-1,self.hps.dimension])
            cv = tf.reshape(tf.tile(tf.expand_dims(prior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            c = tf.concat([cv, cm], axis=-1)
            sample = self.acflow.cond_inverse(x, c, b, m)
            self.sample = tf.reshape(sample, [-1, self.hps.set_size, self.hps.dimension])

            # cd, emd, kde
            if self.hps.lambda_cd > 0:
                cd_loss = chamfer(self.sample, self.x)
                self.loss = self.loss + cd_loss * self.hps.lambda_cd

            if self.hps.lambda_kde > 0:
                density = kde(self.sample, self.x, self.hps.kde_sigma)
                kde_loss = -tf.reduce_mean(density)
                self.loss = self.loss + kde_loss * self.hps.lambda_kde
                self.metric = density

            # compress
            log_likel = self.acflow.cond_forward(x, c, b, m)
            self.log_likel = tf.reshape(log_likel, [-1,self.hps.set_size])