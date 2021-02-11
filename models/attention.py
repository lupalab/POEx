import tensorflow as tf


def uniform_attention(q, v):
    """Uniform attention. Equivalent to np.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      v: values. tensor of shape [B,n,d_v].

    Returns:
      tensor of shape [B,m,d_v].
    """
    total_points = tf.shape(q)[1]
    rep = tf.reduce_mean(v, axis=1, keepdims=True)  # [B,1,d_v]
    rep = tf.tile(rep, [1, total_points, 1])
    return rep


def laplace_attention(q, k, v, scale, normalise):
    """Computes laplace exponential attention.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      scale: float that scales the L1 distance.
      normalise: Boolean that determines whether weights sum to 1.

    Returns:
      tensor of shape [B,m,d_v].
    """
    k = tf.expand_dims(k, axis=1)  # [B,1,n,d_k]
    q = tf.expand_dims(q, axis=2)  # [B,m,1,d_k]
    unnorm_weights = - tf.abs((k - q) / scale)  # [B,m,n,d_k]
    unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [B,m,n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        def weight_fn(x): return 1 + tf.tanh(x)
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
    return rep


def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      normalise: Boolean that determines whether weights sum to 1.

    Returns:
      tensor of shape [B,m,d_v].
    """
    d_k = tf.shape(q)[-1]
    scale = tf.sqrt(tf.cast(d_k, tf.float32))
    unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale  # [B,m,n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = tf.sigmoid
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
    return rep


def multihead_attention(q, k, v, num_heads=4):
    """Computes multi-head attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      num_heads: number of heads. Should divide d_v.

    Returns:
      tensor of shape [B,m,d_v].
    """
    d_k = q.get_shape().as_list()[-1]
    d_v = v.get_shape().as_list()[-1]
    head_size = d_v / num_heads
    key_initializer = tf.random_normal_initializer(stddev=d_k**-0.5)
    value_initializer = tf.random_normal_initializer(stddev=d_v**-0.5)
    rep = tf.constant(0.0)
    for h in range(num_heads):
        o = dot_product_attention(
            tf.layers.conv1d(q, head_size, 1, kernel_initializer=key_initializer,
                             name=f'wq{h}', use_bias=False, padding='VALID'),
            tf.layers.conv1d(k, head_size, 1, kernel_initializer=key_initializer,
                             name=f'wk{h}', use_bias=False, padding='VALID'),
            tf.layers.conv1d(v, head_size, 1, kernel_initializer=key_initializer,
                             name=f'wv{h}', use_bias=False, padding='VALID'),
            normalise=True)
        rep += tf.layers.conv1d(o, d_v, 1, kernel_initializer=value_initializer,
                                name=f'wo{h}', use_bias=False, padding='VALID')
    return rep


class Attention(object):
    """The Attention module."""

    def __init__(self, att_type='multihead', scale=1., normalise=True, num_heads=4, name='attention'):
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pair
        to output an aggregated representation of the context data.
        Args:
          att_type: type of attention. One of the following:
              ['uniform','laplace','dot_product','multihead']
          scale: scale of attention.
          normalise: Boolean determining whether to:
              1. apply softmax to weights so that they sum to 1 across context pts or
              2. apply custom transformation to have weights in [0,1].
          num_heads: number of heads for multihead.
        """
        self._type = att_type
        self._scale = scale
        self._normalise = normalise
        self._num_heads = num_heads
        self._name = name

    def __call__(self, q, k, v):
        """Apply attention to create aggregated representation of v.

        Args:
          q: tensor of shape [B,n1,d_x].
          k: tensor of shape [B,n2,d_x].
          v: tensor of shape [B,n2,d_y].

        Returns:
          tensor of shape [B,n1,d_y]

        """
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            if self._type == 'uniform':
                rep = uniform_attention(q, v)
            elif self._type == 'laplace':
                rep = laplace_attention(q, k, v, self._scale, self._normalise)
            elif self._type == 'dot_product':
                rep = dot_product_attention(q, k, v, self._normalise)
            elif self._type == 'multihead':
                rep = multihead_attention(q, k, v, self._num_heads)
            else:
                raise NameError(
                    "'att_type' not among ['uniform','laplace','dot_product','multihead']")

        return rep
