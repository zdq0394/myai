import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.constant([0.9, 0.85], shape=[1, 2])
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=31), name="w1")
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=31), name="w2")

b1 = tf.nn.relu(tf.Variable(tf.zeros([1,3])))
b2 = tf.nn.relu(tf.Variable(tf.ones([1])))

init_op = tf.global_variables_initializer()

a = tf.matmul(x, w1) + b1
y = tf.matmul(a, w2) + b2

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))
