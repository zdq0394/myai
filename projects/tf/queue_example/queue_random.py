import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

Queue = tf.queue.RandomShuffleQueue(5, 2, "int32")
queue_init = Queue.enqueue_many(([10, 20, 30, 40, 50],))
a = Queue.dequeue()
b = a + 1

Queue_en = Queue.enqueue([b])

with tf.Session() as sess:
    queue_init.run()
    for i in range(10):
        v, _ = sess.run([a, Queue_en])
        print(v)