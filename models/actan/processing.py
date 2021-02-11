import tensorflow as tf

def preprocess(x, b, m):
    x_o = x * m * b
    x_u = x * m * (1-b)
    query = m * (1-b)
    ind = tf.contrib.framework.argsort(query, axis=1, direction='DESCENDING', stable=True)
    x_u = tf.batch_gather(x_u, ind)

    return x_u, x_o

def postprocess(x_u, x, b, m):
    query = m * (1-b)
    ind = tf.contrib.framework.argsort(query, axis=1, direction='DESCENDING', stable=True)
    ind = tf.contrib.framework.argsort(ind, axis=1, direction='ASCENDING', stable=True)
    x_u = tf.batch_gather(x_u, ind)
    sam = x_u * query + x * (1-query)

    return sam
