import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


g1 = tf.Graph()

with g1.as_default():
    a = tf.get_variable("a", [2], initializer=tf.ones_initializer())
    b = tf.get_variable("b", [2], initializer=tf.zeros_initializer())

g2 = tf.Graph()

with g2.as_default():
    a = tf.get_variable("a", [2], initializer=tf.zeros_initializer())
    b = tf.get_variable("b", [2], initializer=tf.ones_initializer())

print("graph1:")
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("a")))
        print(sess.run(tf.get_variable("b")))
print("graph2:")
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("a")))
        print(sess.run(tf.get_variable("b")))