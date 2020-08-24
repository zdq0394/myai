import os
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = \
    keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28)/255.0
test_images = test_images[:1000].reshape(-1, 28*28)/255.0

def create_model():
    model = keras.models.Sequential(
        [
            keras.layers.Dense(512, activation="relu", input_shape=(28*28,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10)
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model

model = create_model()
print(model.summary())

model.fit(
    train_images,
    train_labels,
    epochs=10,
)
checkpoint_path = "../models/keras_model_hdf5/model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir, mode = os.O_CREAT)
model.save(checkpoint_path)

new_model = tf.keras.models.load_model(checkpoint_path)
print(new_model.summary())
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))