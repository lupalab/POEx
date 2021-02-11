import tensorflow as tf

from .transforms import Transform
from .likelihood import Prior
from .processing import preprocess, postprocess

class Flow(object):
    def __init__(self, hps, scope=None):
        self.hps = hps

        self.build(scope)

    def build(self, scope):
        trans_scope = scope + '_trans' if scope else 'trans'
        self.trans = Transform(trans_scope, self.hps)
        prior_scope = scope + '_prior' if scope else 'prior'
        self.prior = Prior(prior_scope, self.hps)

    def forward(self, x, b, m):
        x_u, x_o = preprocess(x, b, m)
        z_u, logdet = self.trans.forward(x_u, x_o, b, m)
        prior_ll = self.prior.logp(z_u, x_o, b, m)
        log_likel = prior_ll + logdet

        return log_likel

    def inverse(self, x, b, m):
        _, x_o = preprocess(x, b, m)
        z_u = self.prior.sample(x_o, b, m)
        x_u, _ = self.trans.inverse(z_u, x_o, b, m)
        x_sam = postprocess(x_u, x, b, m)

        return x_sam

    def mean(self, x, b, m):
        _, x_o = preprocess(x, b, m)
        z_u = self.prior.mean(x_o, b, m)
        x_u, _ = self.trans.inverse(z_u, x_o, b, m)
        x_mean = postprocess(x_u, x, b, m)

        return x_mean

    def cond_forward(self, x, c, b, m):
        x_u, x_o = preprocess(x, b, m)
        c = tf.concat([x_o, c], axis=1)
        z_u, logdet = self.trans.forward(x_u, c, b, m)
        prior_ll = self.prior.logp(z_u, c, b, m)
        log_likel = prior_ll + logdet

        return log_likel

    def cond_inverse(self, x, c, b, m):
        _, x_o = preprocess(x, b, m)
        c = tf.concat([x_o, c], axis=1)
        z_u = self.prior.sample(c, b, m)
        x_u, _ = self.trans.inverse(z_u, c, b, m)
        x_sam = postprocess(x_u, x, b, m)

        return x_sam

    def cond_mean(self, x, c, b, m):
        _, x_o = preprocess(x, b, m)
        c = tf.concat([x_o, c], axis=1)
        z_u = self.prior.mean(c, b, m)
        x_u, _ = self.trans.inverse(z_u, c, b, m)
        x_mean = postprocess(x_u, x, b, m)

        return x_mean


