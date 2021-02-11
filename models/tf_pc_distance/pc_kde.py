import tensorflow as tf

def kde(pred, gt, h):
    N = tf.shape(pred)[1]
    d = tf.shape(pred)[2]
    gt = tf.expand_dims(gt, axis=2)  # [B,N,1,3]
    pred = tf.expand_dims(pred, axis=1)  # [B,1,N,3]
    inp = (gt - pred) / h  # [B,N,N,3]
    kernel = tf.distributions.Normal(loc=0., scale=1.)
    log_kx = tf.reduce_sum(kernel.log_prob(inp), axis=-1)  # [B,N,N]
    log_px = tf.reduce_logsumexp(log_kx, axis=-1)  # [B,N]
    log_likel = log_px - tf.log(tf.cast(N, tf.float32)) - tf.log(h) * tf.cast(d, tf.float32)
    log_likel = tf.reduce_sum(log_likel, axis=-1)  # [B]

    return log_likel