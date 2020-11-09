import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="reminder")

init_op = tf.global_variables_initializer()

loss = tf.square(Y - tf.multiply(X, w) - b)

op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape)*0.33 + 10

with tf.Session("grpc://127.0.0.1:2223") as sess:
    sess.run(init_op)
    for i in range(10):
        for (x, y) in zip(train_X, train_Y):
            sess.run(op, feed_dict={X:x, Y:y})
    print(sess.run(w))
    print(sess.run(b))