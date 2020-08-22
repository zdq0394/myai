import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([3.0, 4.0], name="b")
result = a+b
print("It is a tensor representation:")
print(result)
