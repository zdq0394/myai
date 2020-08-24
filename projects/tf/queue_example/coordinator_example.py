import tensorflow.compat.v1 as tf
import numpy as np
import threading
import time
tf.disable_v2_behavior()

def thread_op(coordinator, thread_id):
    while coordinator.should_stop() == False:
        if np.random.rand() < 0.1:
            print("Stopping from thread_id: %d \n" % thread_id)
            coordinator.request_stop()
        else:
            print("Working on thread_id: %d \n" % thread_id)
        time.sleep(10)

coordinator = tf.train.Coordinator()
threads = [threading.Thread(target= thread_op, args=(coordinator, i)) for i in range(5)]

for j in threads:
    j.start()

coordinator.join(threads)