import tensorflow as tf

tx = tf.constant(
    [[3, 2, 1, 4],
    [4, 3, 1, 2],
    [3, 4, 2, 1],
    [2, 1, 4, 3]],
)

graph = tf.constant(
    [[0.1, 0.2, 0.3, 0.4],
    [0.4, 0.3, 0.1, 0.2],
    [0.3, 0.4, 0.2, 0.1],
    [0.2, 0.1, 0.4, 0.3]],
)

k = 2
batch_size = 4

row = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, k])

# row = tf.reshape(row, [-1])
_, col = tf.nn.top_k(tx, 2)
col = tf.reshape(col, (4,2))

idx = tf.stack([row, col], axis=2)

res = tf.gather_nd(graph, idx)

with tf.Session() as sess:
    r, result = sess.run([idx, res])
    print(result)
    print(result.shape)
    print(r)
    print(r.shape)