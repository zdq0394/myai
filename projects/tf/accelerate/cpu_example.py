import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = tf.Variable(tf.constant([1.0, 2.0], shape=[2]), name="a")

b = tf.Variable(tf.constant([3.0, 4.0], shape=[2]), name="b")

result = a+b

init_op = tf.initialize_all_variables()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init_op)
    print(result)

