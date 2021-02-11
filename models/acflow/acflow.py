import numpy as np
import tensorflow as tf

from .logits import preprocess, postprocess
from .modules import encoder_spec, decoder_spec
from .utils import standard_normal_ll, standard_normal_sample

class ACFlow(object):
    def __init__(self, hps):
        self.hps = hps

    def forward(self, x, b, m=None, cv=None, cm=None):
        '''
        Args:
            x: data, [B,H,W,C] [uint8]
            b: mask, [B,H,W,C] [uint8] 1: condition
            m: miss, [B,H,W,C] [uint8] 1: not missing
            cv: condV, [B,C] [float32]
            cm: condM, [B,H,W,C] [float32]
        Returns:
            log_likel: [B]
        '''
        b = tf.cast(b, tf.float32)
        m = tf.cast(m, tf.float32) if m is not None else tf.ones_like(b)
        r = (1.- b) * m # 1: query
        x, logdet = preprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        z, ldet = encoder_spec(x, b, m, self.hps, self.hps.n_scale,
                               use_batch_norm=False, condV=cv, condM=cm)
        ldet = tf.reduce_sum(ldet, [1, 2, 3])
        logdet += ldet
        prior_ll = standard_normal_ll(z)
        prior_ll = tf.reduce_sum(prior_ll * r, [1, 2, 3])
        log_likel = prior_ll + logdet

        return log_likel

    def inverse(self, x, b, m=None, cv=None, cm=None):
        b = tf.cast(b, tf.float32)
        m = tf.cast(m, tf.float32) if m is not None else tf.ones_like(b)
        r = (1.- b) * m # 1: query
        x, _ = preprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        z = standard_normal_sample(tf.shape(x))
        x = z * r + x * (1. - r)
        x, _ = decoder_spec(x, b, m, self.hps, self.hps.n_scale,
                            use_batch_norm=False, condV=cv, condM=cm)
        x, _ = postprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        
        return x

    def pseudo_mean(self, x, b, m=None, cv=None, cm=None):
        b = tf.cast(b, tf.float32)
        m = tf.cast(m, tf.float32) if m is not None else tf.ones_like(b)
        r = (1.- b) * m # 1: query
        x, _ = preprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        z = tf.zeros(tf.shape(x))
        x = z * r + x * (1. - r)
        x, _ = decoder_spec(x, b, m, self.hps, self.hps.n_scale,
                            use_batch_norm=False, condV=cv, condM=cm)
        x, _ = postprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        
        return x