import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

meta_graph = tf.train.import_meta_graph("../models/model_example1/model2.ckpt.meta")

with tf.Session() as sess:
    meta_graph.restore(sess, "../models/model_example1/model2.ckpt")
    r = sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))
    print(r)