import tensorflow as tf
tfk = tf.keras
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

class Prior(object):
    def __init__(self, name, hps):
        self.name = name or 'prior'
        self.hps = hps

        self.build()

    def build(self):
        if self.hps.prior == 'gaussian':
            self.prior = DiagGaussian(self.name, self.hps)
        elif self.hps.prior == 'mix_gaussian':
            self.prior = MixGaussian(self.name, self.hps)
        elif self.hps.prior == 'autoreg':
            self.prior = AutoReg(self.name, self.hps)
        else:
            raise Exception()

    def logp(self, z, c, b, m):
        return self.prior.logp(z, c, b, m)

    def sample(self, c, b, m):
        return self.prior.sample(c, b, m)

    def mean(self, c, b, m):
        return self.prior.mean(c, b, m)

class DiagGaussian(object):
    def __init__(self, name, hps):
        self.name = name
        self.hps = hps

        self.build()

    def build(self):
        d = self.hps.dimension
        self.net = tfk.Sequential(name='diag_gaussian')
        for i, h in enumerate(self.hps.prior_hids):
            self.net.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
        self.net.add(tfk.layers.Dense(d*2, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))

    def get_dist(self, c, b, m):
        h = tf.concat([c, b, m], axis=1)
        params = self.net(h)
        mean, logs = tf.split(params, 2, axis=1)
        # reorder
        query = m * (1-b)
        order = tf.contrib.framework.argsort(query, direction='DESCENDING', stable=True)
        t = tf.batch_gather(tf.matrix_diag(query), order)
        t = tf.transpose(t, perm=[0,2,1])
        mean = tf.einsum('nd,ndi->ni', mean, t)
        logs = tf.einsum('nd,ndi->ni', logs, t)

        dist = tfd.Normal(mean, tf.exp(logs))

        return dist

    def logp(self, z, c, b, m):
        dist = self.get_dist(c, b, m)
        log_likel = dist.log_prob(z)
        # mask out observed
        query = m * (1-b)
        mask = tf.contrib.framework.sort(query, axis=1, direction='DESCENDING')
        log_likel = tf.reduce_sum(log_likel * mask, axis=1)

        return log_likel 

    def sample(self, c, b, m):
        dist = self.get_dist(c, b, m)

        return dist.sample()

    def mean(self, c, b, m):
        dist = self.get_dist(c, b, m)

        return dist.mean()

class MixGaussian(object):
    def __init__(self, name, hps):
        self.name = name
        self.hps = hps

        self.build()

    def build(self):
        self.net = tfk.Sequential(name='mix_gaussian')
        for i, h in enumerate(self.hps.prior_hids):
            self.net.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
        self.net.add(tfk.layers.Dense(self.hps.n_components * 3, name=f'l{i+1}'))

    def get_params(self, c, b, m):
        d = self.hps.dimension
        h = tf.concat([c, b, m], axis=1)
        params = self.net(h)
        params = tf.tile(tf.expand_dims(params, axis=1), [1,d,1])

        return params

    def logp(self, z, c, b, m):
        params = self.get_params(c, b, m)
        log_likel = mixture_likelihoods(params, z)
        # mask out observed
        query = m * (1-b)
        mask = tf.contrib.framework.sort(query, axis=1, direction='DESCENDING')
        log_likel = tf.reduce_sum(log_likel * mask, axis=1)

        return log_likel

    def sample(self, c, b, m):
        params = self.get_params(c, b, m)
        z = mixture_sample(params)

        return z

    def mean(self, c, b, m):
        params = self.get_params(c, b, m)
        z = mixture_mean(params)

        return z

class AutoReg(object):
    def __init__(self, name, hps):
        self.name = name
        self.hps = hps

        self.build()

    def build(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.rnn_cell = tfk.layers.StackedRNNCells([
                tfk.layers.GRUCell(self.hps.prior_units)
                for _ in range(self.hps.prior_layers)
            ], name='rnn_cell')

            self.rnn_out = tfk.Sequential(name='rnn_out')
            for i, h in enumerate(self.hps.prior_hids):
                self.rnn_out.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.rnn_out.add(tfk.layers.Dense(self.hps.n_components * 3, name=f'l{i+1}'))

    def logp(self, z, c, b, m):
        B = tf.shape(z)[0]
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            state = self.rnn_cell.get_initial_state(batch_size=B, dtype=tf.float32)
            z_t = -tf.ones((B,1), dtype=tf.float32)
            p_list = []
            for t in range(d):
                inp = tf.concat([z_t, c, b, m], axis=1)
                h_t, state = self.rnn_cell(inp, state)
                h_t = tf.concat([h_t, c, b, m], axis=1)
                p_t = self.rnn_out(h_t)
                p_list.append(p_t)
                z_t = tf.expand_dims(z[:,t], axis=1)
            params = tf.stack(p_list, axis=1) # [B,d,n*3]
            log_likel = mixture_likelihoods(params, z)
            # mask out observed
            query = m * (1-b)
            mask = tf.contrib.framework.sort(query, axis=1, direction='DESCENDING')
            log_likel = tf.reduce_sum(log_likel * mask, axis=1)

        return log_likel
        
    def sample(self, c, b, m):
        B = tf.shape(c)[0]
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            state = self.rnn_cell.get_initial_state(batch_size=B, dtype=tf.float32)
            z_t = -tf.ones((B,1), dtype=tf.float32)
            z_list = []
            for t in range(d):
                inp = tf.concat([z_t, c, b, m], axis=1)
                h_t, state = self.rnn_cell(inp, state)
                h_t = tf.concat([h_t, c, b, m], axis=1)
                p_t = self.rnn_out(h_t)
                z_t = mixture_sample_dim(p_t)
                z_list.append(z_t)
            z = tf.concat(z_list, axis=1)

        return z

    def mean(self, c, b, m):
        B = tf.shape(c)[0]
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            state = self.rnn_cell.get_initial_state(batch_size=B, dtype=tf.float32)
            z_t = -tf.ones((B,1), dtype=tf.float32)
            z_list = []
            for t in range(d):
                inp = tf.concat([z_t, c, b, m], axis=1)
                h_t, state = self.rnn_cell(inp, state)
                h_t = tf.concat([h_t, c, b, m], axis=1)
                p_t = self.rnn_out(h_t)
                z_t = mixture_mean_dim(p_t)
                z_list.append(z_t)
            z = tf.concat(z_list, axis=1)

        return z

def mixture_likelihoods(params, targets, base_distribution='gaussian'):
    '''
    Args:
        params: [B,d,c*3]
        targets: [B,d]
    Return:
        log_likelihood: [B,d]
    '''
    targets = tf.expand_dims(targets, axis=-1)
    logits, means, lsigmas = tf.split(params, 3, axis=2)
    sigmas = tf.exp(lsigmas)
    if base_distribution == 'gaussian':
        log_norm_consts = -lsigmas - 0.5 * np.log(2.0 * np.pi)
        log_kernel = -0.5 * tf.square((targets - means) / sigmas)
    elif base_distribution == 'laplace':
        log_norm_consts = -lsigmas - np.log(2.0)
        log_kernel = -tf.abs(targets - means) / sigmas
    elif base_distribution == 'logistic':
        log_norm_consts = -lsigmas
        diff = (targets - means) / sigmas
        log_kernel = -tf.nn.softplus(diff) - tf.nn.softplus(-diff)
    else:
        raise NotImplementedError
    log_exp_terms = log_kernel + log_norm_consts + logits
    log_likelihoods = tf.reduce_logsumexp(log_exp_terms, -1) - tf.reduce_logsumexp(logits, -1)
    
    return log_likelihoods

def mixture_sample_dim(params_dim, base_distribution='gaussian'):
    '''
    Args:
        params_dim: [B,n*3]
    Return:
        samp: [B,1]
    '''
    B = tf.shape(params_dim)[0]
    logits, means, lsigmas = tf.split(params_dim, 3, axis=1)
    sigmas = tf.exp(lsigmas)
    # sample multinomial
    js = tf.multinomial(logits, 1)  # int64
    inds = tf.concat([tf.expand_dims(tf.range(tf.cast(B,tf.int64), dtype=tf.int64), -1), js], axis=1)
    # Sample from base distribution.
    if base_distribution == 'gaussian':
        zs = tf.random_normal((B, 1))
    elif base_distribution == 'laplace':
        zs = tf.log(tf.random_uniform((B, 1))) - \
            tf.log(tf.random_uniform((B, 1)))
    elif base_distribution == 'logistic':
        x = tf.random_uniform((B, 1))
        zs = tf.log(x) - tf.log(1.0 - x)
    else:
        raise NotImplementedError()
    # scale and shift
    mu_zs = tf.expand_dims(tf.gather_nd(means, inds), axis=-1)
    sigma_zs = tf.expand_dims(tf.gather_nd(sigmas, inds), axis=-1)
    samp = sigma_zs * zs + mu_zs
    
    return samp

def mixture_mean_dim(params_dim, base_distribution='gaussian'):
    logits, means, lsigmas = tf.split(params_dim, 3, axis=1)
    weights = tf.nn.softmax(logits, axis=-1)

    return tf.reduce_sum(weights * means, axis=1, keepdims=True)

def mixture_sample(params, base_distribution='gaussian'):
    '''
    Args:
        params: [B,d,n*3]
    Return:
        samp: [B,d]
    '''
    B = tf.shape(params)[0]
    d = tf.shape(params)[1]
    logits, means, lsigmas = tf.split(params, 3, axis=2)
    sigmas = tf.exp(lsigmas)
    # sample multinomial
    ind0 = tf.tile(tf.reshape(tf.range(B), [B,1,1]), [1,d,1])
    ind1 = tf.tile(tf.reshape(tf.range(d), [1,d,1]), [B,1,1])
    ind2 = tf.expand_dims(tfd.Categorical(logits).sample(), -1)  # int32
    inds = tf.concat([ind0,ind1,ind2], axis=2)
    # Sample from base distribution.
    if base_distribution == 'gaussian':
        zs = tf.random_normal((B, d))
    elif base_distribution == 'laplace':
        zs = tf.log(tf.random_uniform((B, d))) - \
            tf.log(tf.random_uniform((B, d)))
    elif base_distribution == 'logistic':
        x = tf.random_uniform((B, d))
        zs = tf.log(x) - tf.log(1.0 - x)
    else:
        raise NotImplementedError()
    # scale and shift
    mu_zs = tf.gather_nd(means, inds)
    sigma_zs = tf.gather_nd(sigmas, inds)
    samp = sigma_zs * zs + mu_zs
    
    return samp

def mixture_mean(params, base_distribution='gaussian'):
    logits, means, lsigmas = tf.split(params, 3, axis=2)
    weights = tf.nn.softmax(logits, axis=-1)

    return tf.reduce_sum(weights * means, axis=2)
