import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

worker_0 = "127.0.0.1:2222"
worker_1 = "127.0.0.1:2223"
worker_hosts = [worker_0, worker_1]

cluster_spec = tf.train.ClusterSpec({"worker": worker_hosts})

server = tf.train.Server(cluster_spec, job_name="worker", task_index=1)
server.join()