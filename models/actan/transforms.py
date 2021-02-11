import tensorflow as tf
tfk = tf.keras
import numpy as np

'''
conditional transformations:
x: [B, d]
c: [B, d]
z: [B, d]
b: [B, d] binary mask ==> 1: conditioning
m: [B, d] missing mask ==> 1: not missing
q: query ==> m*(1-b)
'''

# base class
class BaseTransform(object):
    def __init__(self, name, hps):
        self.name = name
        self.hps = hps

        self.build()

    def build(self):
        pass

    def forward(self, x, c, b, m):
        raise NotImplementedError()

    def inverse(self, z, c, b, m):
        raise NotImplementedError()

class Transform(BaseTransform):
    def __init__(self, name, hps):
        name = name or 'transform'
        super(Transform, self).__init__(name, hps)

    def build(self):
        self.modules = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i, name in enumerate(self.hps.transform):
                m = TRANS[name](f'{i}', self.hps)
                self.modules.append(m)

    def forward(self, x, c, b, m):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            logdet = 0.
            for module in self.modules:
                x, ldet = module.forward(x, c, b, m)
                logdet = logdet + ldet

        return x, logdet

    def inverse(self, z, c, b, m):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            logdet = 0.
            for module in reversed(self.modules):
                z, ldet = module.inverse(z, c, b, m)
                logdet = logdet + ldet

        return z, logdet

class TransLayer(BaseTransform):
    def __init__(self, name, hps):
        name = f'layer_{name}'
        super(TransLayer, self).__init__(name, hps)
    
    def build(self):
        self.modules = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            cfg = self.hps.layer_cfg
            for i, name in enumerate(cfg):
                m = TRANS[name](f'{i}', self.hps)
                self.modules.append(m)

    def forward(self, x, c, b, m):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            logdet = 0.
            for module in self.modules:
                x, ldet = module.forward(x, c, b, m)
                logdet = logdet + ldet

        return x, logdet

    def inverse(self, z, c, b, m):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            logdet = 0.
            for module in reversed(self.modules):
                z, ldet = module.inverse(z, c, b, m)
                logdet = logdet + ldet

        return z, logdet

class Reverse(BaseTransform):
    def __init__(self, name, hps):
        name = f'reverse_{name}'
        super(Reverse, self).__init__(name, hps)

    def forward(self, x, c, b, m):
        query = m * (1-b) # [B, d]
        sorted_query = tf.contrib.framework.sort(query, axis=-1, direction='DESCENDING')
        reverse_query = tf.reverse(sorted_query, [-1])
        ind = tf.contrib.framework.argsort(reverse_query, axis=-1, direction='DESCENDING', stable=True)
        z = tf.reverse(x, [-1])
        z = tf.batch_gather(z, ind)
        ldet = 0.0

        return z, ldet

    def inverse(self, z, c, b, m):
        query = m * (1-b) # [B, d]
        sorted_query = tf.contrib.framework.sort(query, axis=-1, direction='DESCENDING')
        reverse_query = tf.reverse(sorted_query, [-1])
        ind = tf.contrib.framework.argsort(reverse_query, axis=-1, direction='DESCENDING', stable=True)
        x = tf.reverse(z, [-1])
        x = tf.batch_gather(x, ind)
        ldet = 0.0

        return x, ldet

class LeakyReLU(BaseTransform):
    def __init__(self, name, hps):
        name = f'lrelu_{name}'
        super(LeakyReLU, self).__init__(name, hps)

    def build(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.alpha = tf.nn.sigmoid(
                tf.get_variable('log_alpha', 
                                initializer=5.0, 
                                dtype=tf.float32))

    def forward(self, x, c, b, m):
        query = m * (1-b) # [B, d]
        sorted_query = tf.contrib.framework.sort(query, axis=-1, direction='DESCENDING')
        num_negative = tf.reduce_sum(tf.cast(tf.less(x, 0.0), tf.float32) * sorted_query, axis=1)
        ldet = num_negative * tf.log(self.alpha)
        z = tf.maximum(x, self.alpha * x)

        return z, ldet

    def inverse(self, z, c, b, m):
        query = m * (1-b) # [B, d]
        sorted_query = tf.contrib.framework.sort(query, axis=-1, direction='DESCENDING')
        num_negative = tf.reduce_sum(tf.cast(tf.less(z, 0.0), tf.float32) * sorted_query, axis=1)
        ldet = -1. * num_negative * tf.log(self.alpha)
        x = tf.minimum(z, z / self.alpha)

        return x, ldet

class Rescale(BaseTransform):
    def __init__(self, name, hps):
        name = f'rescale_{name}'
        super(Rescale, self).__init__(name, hps)

    def build(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            d = self.hps.dimension
            self.logs = tf.get_variable('logs', initializer=tf.zeros((1, d)), dtype=tf.float32)
    
    def forward(self, x, c, b, m):
        B = tf.shape(x)[0]
        query = m * (1-b)
        ind = tf.contrib.framework.argsort(query, axis=-1, direction='DESCENDING', stable=True)
        logs_tiled = tf.tile(self.logs, [B, 1])
        logs = tf.batch_gather(logs_tiled, ind)
        z = tf.multiply(x, tf.exp(logs))
        ldet = tf.reduce_sum(logs_tiled * query, axis=-1)

        return z, ldet

    def inverse(self, z, c, b, m):
        B = tf.shape(z)[0]
        query = m * (1-b)
        ind = tf.contrib.framework.argsort(query, axis=-1, direction='DESCENDING', stable=True)
        logs_tiled = tf.tile(self.logs, [B, 1])
        logs = tf.batch_gather(logs_tiled, ind)
        x = tf.divide(z, tf.exp(logs))
        ldet = -1 * tf.reduce_sum(logs_tiled * query, axis=-1)

        return x, ldet

class Coupling1(BaseTransform):
    def __init__(self, name, hps):
        name = f'cp1_{name}'
        super(Coupling1, self).__init__(name, hps)

    def build(self):
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.net1 = tfk.Sequential(name=f'{self.name}/ms1')
            for i, h in enumerate(self.hps.coupling_hids):
                self.net1.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.net1.add(tfk.layers.Dense(d, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))

            self.net2 = tfk.Sequential(name=f'{self.name}/ms2')
            for i, h in enumerate(self.hps.coupling_hids):
                self.net2.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.net2.add(tfk.layers.Dense(d, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))

        self.nets = {1: self.net1, 2: self.net2}
    
    def get_params(self, x, c, b, m, id):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        mask = np.arange(d, dtype=np.float32)
        mask = tf.mod(mask + id, 2)
        mask = tf.tile(tf.expand_dims(mask, axis=0), [B,1])
        inp = tf.concat([x*mask, mask, c, b, m], axis=1)
        shift = self.nets[id](inp)
        # reorder
        query = m * (1-b)
        order = tf.contrib.framework.argsort(query, direction='DESCENDING', stable=True)
        t = tf.batch_gather(tf.matrix_diag(query), order)
        t = tf.transpose(t, perm=[0,2,1])
        shift = tf.einsum('nd,ndi->ni', shift, t)
        # mask
        shift = shift * (1. - mask)

        return shift

    def forward(self, x, c, b, m):
        B = tf.shape(x)[0]
        ldet = tf.zeros(B, dtype=tf.float32)
        # part 1
        shift = self.get_params(x, c, b, m, 1)
        x = x + shift
        # part 2
        shift = self.get_params(x, c, b, m, 2)
        x = x + shift

        return x, ldet

    def inverse(self, z, c, b, m):
        B = tf.shape(z)[0]
        ldet = tf.zeros(B, dtype=tf.float32)
        # part 2
        shift = self.get_params(z, c, b, m, 2)
        z = z - shift
        # part 1
        shift = self.get_params(z, c, b, m, 1)
        z = z - shift

        return z, ldet    

class Coupling2(BaseTransform):
    def __init__(self, name, hps):
        name = f'cp2_{name}'
        super(Coupling2, self).__init__(name, hps)

    def build(self):
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.net1 = tfk.Sequential(name=f'{self.name}/ms1')
            for i, h in enumerate(self.hps.coupling_hids):
                self.net1.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.net1.add(tfk.layers.Dense(d*2, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))

            self.net2 = tfk.Sequential(name=f'{self.name}/ms2')
            for i, h in enumerate(self.hps.coupling_hids):
                self.net2.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.net2.add(tfk.layers.Dense(d*2, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))

        self.nets = {1: self.net1, 2: self.net2}

    def get_params(self, x, c, b, m, id):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        mask = np.arange(d, dtype=np.float32)
        mask = tf.mod(mask + id, 2)
        mask = tf.tile(tf.expand_dims(mask, axis=0), [B,1])
        inp = tf.concat([x*mask, mask, c, b, m], axis=1)
        params = self.nets[id](inp)
        scale, shift = tf.split(params, 2, axis=1)
        # reorder
        query = m * (1-b)
        order = tf.contrib.framework.argsort(query, direction='DESCENDING', stable=True)
        t = tf.batch_gather(tf.matrix_diag(query), order)
        t = tf.transpose(t, perm=[0,2,1])
        scale = tf.einsum('nd,ndi->ni', scale, t)
        shift = tf.einsum('nd,ndi->ni', shift, t)
        # mask
        scale = scale * (1. - mask)
        shift = shift * (1. - mask)

        return scale, shift

    def forward(self, x, c, b, m):
        B = tf.shape(x)[0]
        ldet = tf.zeros(B, dtype=tf.float32)
        # part 1
        scale, shift = self.get_params(x, c, b, m, 1)
        x = (x + shift) * tf.exp(scale)
        ldet = ldet + tf.reduce_sum(scale, axis=1)
        # part 2
        scale, shift = self.get_params(x, c, b, m, 2)
        x = (x + shift) * tf.exp(scale)
        ldet = ldet + tf.reduce_sum(scale, axis=1)

        return x, ldet

    def inverse(self, z, c, b, m):
        B = tf.shape(z)[0]
        ldet = tf.zeros(B, dtype=tf.float32)
        # part 2
        scale, shift = self.get_params(z, c, b, m, 2)
        z = z * tf.exp(-scale) - shift
        ldet = ldet - tf.reduce_sum(scale, axis=1)
        # part 1
        scale, shift = self.get_params(z, c, b, m, 1)
        z = z * tf.exp(-scale) - shift
        ldet = ldet - tf.reduce_sum(scale, axis=1)

        return z, ldet

class RNNCoupling1(BaseTransform):
    def __init__(self, name, hps):
        name = f'rnncp1_{name}'
        super(RNNCoupling1, self).__init__(name, hps)

    def build(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.rnn_cell = tfk.layers.StackedRNNCells([
                tfk.layers.GRUCell(self.hps.rnncp_units)
                for _ in range(self.hps.rnncp_layers)
            ], name='rnn_cell')

            self.rnn_out = tfk.layers.Dense(1, name='rnn_out')

    def forward(self, x, c, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            state = self.rnn_cell.get_initial_state(batch_size=B, dtype=tf.float32)
            x_t = -tf.ones((B,1), dtype=tf.float32)
            z_list = []
            ldet =0.0
            for t in range(d):
                inp = tf.concat([x_t, c, b, m], axis=1)
                h_t, state = self.rnn_cell(inp, state)
                m_t = self.rnn_out(h_t)
                x_t = tf.expand_dims(x[:, t], axis=1)
                z_t = x_t + m_t
                z_list.append(z_t)
            z = tf.concat(z_list, axis=1)
        
        return z, ldet

    def inverse(self, z, c, b, m):
        B = tf.shape(z)[0]
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            state = self.rnn_cell.get_initial_state(batch_size=B, dtype=tf.float32)
            x_t = -tf.ones((B,1), dtype=tf.float32)
            x_list = []
            ldet = 0.0
            for t in range(d):
                inp = tf.concat([x_t, c, b, m], axis=1)
                h_t, state = self.rnn_cell(inp, state)
                m_t = self.rnn_out(h_t)
                z_t = tf.expand_dims(z[:, t], axis=1)
                x_t = z_t - m_t
                x_list.append(x_t)
            x = tf.concat(x_list, axis=1)

        return x, ldet

class RNNCoupling2(BaseTransform):
    def __init__(self, name, hps):
        name = f'rnncp2_{name}'
        super(RNNCoupling2, self).__init__(name, hps)

    def build(self):
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.rnn_cell = tfk.layers.StackedRNNCells([
                tfk.layers.GRUCell(self.hps.rnncp_units)
                for _ in range(self.hps.rnncp_layers)
            ], name='rnn_cell')

            self.rnn_out = tfk.layers.Dense(2, name='rnn_out')

            self.rescale = tf.get_variable('rescale', [1], initializer=tf.zeros_initializer())

    def forward(self, x, c, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            state = self.rnn_cell.get_initial_state(batch_size=B, dtype=tf.float32)
            x_t = -tf.ones((B,1), dtype=tf.float32)
            z_list = []
            ldet = tf.zeros((B,), dtype=tf.float32)
            for t in range(d):
                inp = tf.concat([x_t, c, b, m], axis=1)
                h_t, state = self.rnn_cell(inp, state)
                p_t = self.rnn_out(h_t)
                s_t, m_t = tf.split(p_t, 2, axis=1)
                s_t = self.rescale * tf.tanh(s_t)
                x_t = tf.expand_dims(x[:, t], axis=1)
                z_t = (x_t + m_t) * tf.exp(s_t)
                z_list.append(z_t)
                ldet += tf.reduce_sum(s_t, axis=1)
            z = tf.concat(z_list, axis=1)
        
        return z, ldet

    def inverse(self, z, c, b, m):
        B = tf.shape(z)[0]
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            state = self.rnn_cell.get_initial_state(batch_size=B, dtype=tf.float32)
            x_t = -tf.ones((B,1), dtype=tf.float32)
            x_list = []
            ldet = tf.zeros((B,), dtype=tf.float32)
            for t in range(d):
                inp = tf.concat([x_t, c, b, m], axis=1)
                h_t, state = self.rnn_cell(inp, state)
                p_t = self.rnn_out(h_t)
                s_t, m_t = tf.split(p_t, 2, axis=1)
                s_t = self.rescale * tf.tanh(s_t)
                z_t = tf.expand_dims(z[:, t], axis=1)
                x_t = z_t * tf.exp(-s_t) - m_t
                x_list.append(x_t)
                ldet -= tf.reduce_sum(s_t, axis=1)
            x = tf.concat(x_list, axis=1)

        return x, ldet

class Linear(BaseTransform):
    def __init__(self, name, hps):
        name = f'linear_{name}'
        super(Linear, self).__init__(name, hps)

    def build(self):
        d = self.hps.dimension
        r = self.hps.linear_rank
        r = d if r <= 0 else r
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            np_w = np.eye(d).astype("float32")
            self.w = tf.get_variable('W', initializer=np_w)
            self.b = tf.get_variable('b', initializer=tf.zeros([d]))

            self.wnn = tfk.Sequential(name=f'{self.name}/wnn')
            self.bnn = tfk.Sequential(name=f'{self.name}/bnn')
            for i, h in enumerate(self.hps.linear_hids):
                self.wnn.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
                self.bnn.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.wnn.add(tfk.layers.Dense(2*d*r, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))
            self.bnn.add(tfk.layers.Dense(d, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))

    def get_params(self, c, b, m):
        B = tf.shape(c)[0]
        d = self.hps.dimension
        r = self.hps.linear_rank
        r = d if r <= 0 else r
        h = tf.concat([c, b, m], axis=1)
        wc = self.wnn(h)
        wc1, wc2 = tf.split(wc, 2, axis=1)
        wc1 = tf.reshape(wc1, [B,d,r])
        wc2 = tf.reshape(wc2, [B,r,d])
        wc = tf.matmul(wc1, wc2)
        bc = self.bnn(h)
        weight = wc + self.w
        bias = bc + self.b
        # reorder
        query = m * (1-b)
        order = tf.contrib.framework.argsort(query, direction='DESCENDING', stable=True)
        t = tf.batch_gather(tf.matrix_diag(query), order)
        weight = tf.matmul(tf.matmul(t, weight), tf.transpose(t, perm=[0,2,1]))
        bias = tf.squeeze(tf.matmul(t, tf.expand_dims(bias, axis=-1)), axis=-1)
        # add a diagnal part
        diag = tf.matrix_diag(tf.contrib.framework.sort(1-query, direction='ASCENDING'))
        weight += diag
        
        return weight, bias

    def forward(self, x, c, b, m):
        weight, bias = self.get_params(c, b, m)
        weight = tf.cast(weight, tf.float64)
        ldet = tf.cast(tf.linalg.logdet(weight), tf.float32)
        weight = tf.cast(weight, tf.float32)
        z = tf.einsum('ai,aik->ak', x, weight) + bias

        return z, ldet

    def inverse(self, z, c, b, m):
        weight, bias = self.get_params(c, b, m)
        weight = tf.cast(weight, tf.float64)
        ldet = -1 * tf.cast(tf.linalg.logdet(weight), tf.float32)
        W_inv = tf.cast(tf.linalg.inv(weight), tf.float32)
        x = tf.einsum('ai,aik->ak', z-bias, W_inv)

        return x, ldet

class LULinear(BaseTransform):
    def __init__(self, name, hps):
        name = f'linear_{name}'
        super(LULinear, self).__init__(name, hps)

    def build(self):
        d = self.hps.dimension
        r = self.hps.linear_rank
        r = d if r <= 0 else r
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            np_w = np.eye(d).astype("float32")
            self.w = tf.get_variable('W', initializer=np_w)
            self.b = tf.get_variable('b', initializer=tf.zeros([d]))

            self.wnn = tfk.Sequential(name=f'{self.name}/wnn')
            self.bnn = tfk.Sequential(name=f'{self.name}/bnn')
            for i, h in enumerate(self.hps.linear_hids):
                self.wnn.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
                self.bnn.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.wnn.add(tfk.layers.Dense(2*d*r, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))
            self.bnn.add(tfk.layers.Dense(d, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))

    def get_params(self, c, b, m):
        B = tf.shape(c)[0]
        d = self.hps.dimension
        r = self.hps.linear_rank
        r = d if r <= 0 else r
        h = tf.concat([c, b, m], axis=1)
        wc = self.wnn(h)
        wc1, wc2 = tf.split(wc, 2, axis=1)
        wc1 = tf.reshape(wc1, [B,d,r])
        wc2 = tf.reshape(wc2, [B,r,d])
        wc = tf.matmul(wc1, wc2)
        bc = self.bnn(h)
        weight = wc + self.w
        bias = bc + self.b
        # reorder
        query = m * (1-b)
        order = tf.contrib.framework.argsort(query, direction='DESCENDING', stable=True)
        t = tf.batch_gather(tf.matrix_diag(query), order)
        weight = tf.matmul(tf.matmul(t, weight), tf.transpose(t, perm=[0,2,1]))
        bias = tf.squeeze(tf.matmul(t, tf.expand_dims(bias, axis=-1)), axis=-1)
        
        return weight, bias

    def get_LU(self, W, b, m):
        d = self.hps.dimension
        U = tf.matrix_band_part(W, 0, -1)
        L = tf.eye(d) + W - U
        A = tf.matmul(L, U)
        # add a diagnal part
        query = m * (1-b)
        diag = tf.matrix_diag(tf.contrib.framework.sort(1-query, axis=1, direction='ASCENDING'))
        U += diag

        return A, L, U

    def forward(self, x, c, b, m):
        weight, bias = self.get_params(c, b, m)
        A, L, U = self.get_LU(weight, b, m)
        ldet = tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(U))), axis=1)
        z = tf.einsum('ai,aik->ak', x, A) + bias

        return z, ldet

    def inverse(self, z, c, b, m):
        weight, bias = self.get_params(c, b, m)
        A, L, U = self.get_LU(weight, b, m)
        ldet = -1 * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(U))), axis=1)
        Ut = tf.transpose(U, perm=[0, 2, 1])
        Lt = tf.transpose(L, perm=[0, 2, 1])
        zt = tf.expand_dims(z - bias, -1)
        sol = tf.matrix_triangular_solve(Ut, zt)
        x = tf.matrix_triangular_solve(Lt, sol, lower=False)
        x = tf.squeeze(x, axis=-1)

        return x, ldet

class Affine(BaseTransform):
    def __init__(self, name, hps):
        name = f'affine_{name}'
        super(Affine, self).__init__(name, hps)

    def build(self):
        d = self.hps.dimension
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.net = tfk.Sequential(name=f'{self.name}/ms')
            for i, h in enumerate(self.hps.affine_hids):
                self.net.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.net.add(tfk.layers.Dense(d*2, name=f'l{i+1}', kernel_initializer=tf.zeros_initializer()))

    def get_params(self, c, b, m):
        h = tf.concat([c, b, m], axis=1)
        params = self.net(h)
        shift, scale = tf.split(params, 2, axis=1)
        # reorder
        query = m * (1-b)
        order = tf.contrib.framework.argsort(query, direction='DESCENDING', stable=True)
        t = tf.batch_gather(tf.matrix_diag(query), order)
        t = tf.transpose(t, perm=[0,2,1])
        scale = tf.einsum('nd,ndi->ni', scale, t)
        shift = tf.einsum('nd,ndi->ni', shift, t)

        return shift, scale
    
    def forward(self, x, c, b, m):
        shift, scale = self.get_params(c, b, m)
        z = tf.multiply(x, tf.exp(scale)) + shift
        ldet = tf.reduce_sum(scale, axis=1)

        return z, ldet

    def inverse(self, z, c, b, m):
        shift, scale = self.get_params(c, b, m)
        x = tf.divide(z-shift, tf.exp(scale))
        ldet = -1 * tf.reduce_sum(scale, axis=1)

        return x, ldet

# register all modules
TRANS = {
    'AF': Affine,
    'CP1': Coupling1,
    'CP2': Coupling2,
    'RCP1': RNNCoupling1,
    'RCP2': RNNCoupling2,
    'R': Reverse,
    'S': Rescale,
    'LR': LeakyReLU,
    'L': Linear,
    'ML': LULinear,
    'TL': TransLayer
}

if __name__ == '__main__':
    from pprint import pformat
    from easydict import EasyDict as edict

    hps = edict()
    hps.dimension = 8
    hps.linear_hids = [32,32]
    hps.affine_hids = [32,32]
    hps.rnncp_units = 32
    hps.rnncp_layers = 2
    hps.coupling_hids = [32,32]
    hps.transform = ['CP2']

    x_ph = tf.placeholder(tf.float32, [32,8])
    c_ph = tf.placeholder(tf.float32, [32,8])
    b_ph = tf.placeholder(tf.float32, [32,8])
    m_ph = tf.placeholder(tf.float32, [32,8])
    
    l1 = Transform('1', hps)
    l2 = Transform('2', hps)
    z, fdet1 = l1.forward(x_ph, c_ph, b_ph, m_ph)
    z, fdet2 = l2.forward(z, c_ph, b_ph, m_ph)
    fdet = fdet1 + fdet2

    x, bdet2 = l2.inverse(z, c_ph, b_ph, m_ph)
    x, bdet1 = l1.inverse(x, c_ph, b_ph, m_ph)
    bdet = bdet1 + bdet2

    q = tf.contrib.framework.sort(m_ph * (1-b_ph), direction='DESCENDING')
    err = tf.reduce_sum(tf.square(x_ph - x)*q)
    det = tf.reduce_sum(fdet + bdet)

    loss = tf.reduce_sum(tf.square(z)) - fdet
    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('='*20)
    print('Variables:')
    print(pformat(tf.trainable_variables()))

    for i in range(1000):
        x_nda = np.random.randn(32,8)
        c_nda = np.random.randn(32,8)
        b_nda = (np.random.rand(32, 8) > 0.5).astype(np.float32)
        m_nda = (np.random.rand(32, 8) > 0.5).astype(np.float32)
        b_nda = m_nda * b_nda
        feed_dict = {x_ph:x_nda, c_ph:c_nda, b_ph:b_nda, m_ph:m_nda}

        res = sess.run([err,det], feed_dict)
        print(f'err:{res[0]} det:{res[1]}')
        sess.run(train_op, feed_dict)

