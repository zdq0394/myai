import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

queue = tf.queue.FIFOQueue(100, "float")
enqueue = queue.enqueue([tf.random_normal([10])])

qr = tf.train.QueueRunner(queue, [enqueue]*10)

tf.train.add_queue_runner(qr)

out_tensor = queue.dequeue()

with tf.Session() as sess:
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    for i in range(10):
        print(sess.run(out_tensor))

    coordinator.request_stop()
    coordinator.join(threads)
