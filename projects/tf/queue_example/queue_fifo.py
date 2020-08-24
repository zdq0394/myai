import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

Queue = tf.queue.FIFOQueue(2, "int32")
queue_init = Queue.enqueue_many(([10, 100],))
a = Queue.dequeue()
b = a + 10

Queue_en = Queue.enqueue([b])

with tf.Session() as sess:
    queue_init.run()
    for i in range(10):
        v, _ = sess.run([a, Queue_en])
        print(v)