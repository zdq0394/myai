import tensorflow as tf

data = tf.constant([1,2])
print("Tensor:", data)

print("Array:", data.numpy())

import numpy as np
arr_list = np.arange(0, 100)
shape = arr_list.shape
print(arr_list)
print(shape)

dataset = tf.data.Dataset.from_tensor_slices(arr_list)
dataset_iterator = dataset.shuffle(shape[0]).batch(10)