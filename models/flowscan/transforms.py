import tensorflow as tf
tfk = tf.keras
import numpy as np

# base class
class BaseTransform(object):
    def __init__(self, hps, name='base'):
        self.name = name
        self.hps = hps

        self.build()

    def build(self):
        pass

    def forward(self, x, c):
        raise NotImplementedError()

    def inverse(self, z, c):
        raise NotImplementedError()


class SetTransform(BaseTransform):
    def __init__(self, hps, name='transform'):
        super(SetTransform, self).__init__(hps, name)

    def build(self):
        self.modules = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i, name in enumerate(self.hps.transform):
                m = TRANS[name](self.hps, f'{i}')
                self.modules.append(m)

    def forward(self, x, c):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            logdet = 0.
            for module in self.modules:
                x, ldet = module.forward(x, c)
                logdet = logdet + ldet

        return x, logdet

    def inverse(self, z, c):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            logdet = 0.
            for module in reversed(self.modules):
                z, ldet = module.inverse(z, c)
                logdet = logdet + ldet

        return z, logdet
