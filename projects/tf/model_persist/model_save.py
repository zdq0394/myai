import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = tf.Variable(tf.constant([1.0, 2.0], shape=[2]), name="a")
b = tf.Variable(tf.constant([3.0, 4.0], shape=[2]), name="b")

result = a+b

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "../models/model_example1/model2.ckpt")