import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .base import BaseModel
from .fns_encoder import *
from .flow.transforms import Transform
from .attention import Attention
from .networks import dense_nn

class ACSetFlow(BaseModel):
    def __init__(self, hps):
        self.prior_net = LatentEncoder(hps, name='prior')
        self.posterior_net = LatentEncoder(hps, name='posterior')
        self.self_attn = Attention(name='self_attn')
        self.cross_attn = Attention(name='cross_attn')
        self.generator_net = Generator(hps)
        super(ACSetFlow, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('acset_flow', reuse=tf.AUTO_REUSE):
            self.xc = tf.placeholder(tf.float32, [None, self.hps.set_size, None, self.hps.x_dim])
            self.yc = tf.placeholder(tf.float32, [None, self.hps.set_size, None, self.hps.y_dim])
            self.xt = tf.placeholder(tf.float32, [None, self.hps.set_size, None, self.hps.x_dim])
            self.yt = tf.placeholder(tf.float32, [None, self.hps.set_size, None, self.hps.y_dim])

            # build transform
            self.transform = Transform(self.hps)

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
            prior_inputs = tf.concat([self.xc, self.yc], axis=-1)
            prior = self.prior_net(prior_inputs)
            prior_sample = prior.sample() # [B,d]
            prior_sample, _ = self.transform.inverse(prior_sample)

            # self attention embedding
            ctx_rep = tf.concat([xc, yc], axis=-1)
            ctx_rep = dense_nn(ctx_rep, self.hps.attn_value_hids, self.hps.attn_dim, name='attn_value')
            xt_rep = dense_nn(xt, self.hps.attn_key_hids[:-1], self.hps.attn_key_hids[-1], name='attn_key')
            xc_rep = dense_nn(xc, self.hps.attn_key_hids[:-1], self.hps.attn_key_hids[-1], name='attn_key')
            cs = self.self_attn(xt_rep, xc_rep, ctx_rep) # [B*K, Nt, d]
            cs = tf.reshape(cs, [B,K,Nt,self.hps.attn_dim])

            # cross attention embedding
            ctx_rep = tf.reshape(ctx_rep, [B,K,1,Nc,self.hps.attn_dim])
            ctx_rep = tf.tile(ctx_rep, [1,1,K,1,1])
            ctx_rep = tf.reshape(ctx_rep, [B*K*K,Nc,self.hps.attn_dim])
            xc = tf.tile(tf.reshape(self.xc, [B,K,1,Nc,dx]), [1,1,K,1,1])
            xc = tf.reshape(xc, [B*K*K,Nc,dx])
            xc_rep = dense_nn(xc, self.hps.attn_key_hids[:-1], self.hps.attn_key_hids[-1], name='attn_key')
            xt = tf.tile(tf.reshape(self.xt, [B,1,K,Nt,dx]), [1,K,1,1,1])
            xt = tf.reshape(xt, [B*K*K,Nt,dx])
            xt_rep = dense_nn(xt, self.hps.attn_key_hids[:-1], self.hps.attn_key_hids[-1], name='attn_key')
            cc = self.cross_attn(xt_rep, xc_rep, ctx_rep) # [B*K*K,Nt,d]
            cc = tf.reshape(cc, [B,K,K,Nt,self.hps.attn_dim])
            mask = tf.reshape(1.-tf.eye(K), [1,K,K,1,1])
            cc = tf.reduce_sum(cc*mask, axis=1) / tf.cast(K-1, tf.float32)

            # posterior
            posterior_inputs = tf.concat([self.xt, self.yt], axis=-1)
            posterior = self.posterior_net(posterior_inputs)
            posterior_sample = posterior.sample() #[B,d]

            # kl term
            z_sample, logdet = self.transform.forward(posterior_sample)
            logp = tf.reduce_sum(prior.log_prob(z_sample), axis=1) + logdet
            kl = tf.reduce_sum(posterior.entropy(), axis=1) + logp #[B]

            # generator p(yt | xt, z)
            cz = tf.tile(tf.reshape(posterior_sample, [B,1,1,self.hps.latent_dim]), [1,K,Nt,1])
            generator_inputs = tf.concat([self.xt, cz, cs, cc], axis=-1)
            generator_dist = self.generator_net(generator_inputs)
            log_likel = generator_dist.log_prob(self.yt) # [B,K,Nt]
            log_likel = tf.reduce_mean(log_likel, axis=(1,2))
            tf.summary.scalar('log_likel', tf.reduce_mean(log_likel))

            # elbo
            self.elbo = log_likel + kl / tf.cast(K*Nt, tf.float32)
            self.metric = self.elbo
            self.loss = tf.reduce_mean(-self.elbo)
            tf.summary.scalar('loss', self.loss)

            # sample
            cz = tf.tile(tf.reshape(prior_sample, [B,1,1,self.hps.latent_dim]), [1,K,Nt,1])
            sampler_inputs = tf.concat([self.xt, cz, cs, cc], axis=-1)
            sampler_dist = self.generator_net(sampler_inputs)
            self.sample = sampler_dist.sample() #[B,K,Nt,dy]
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