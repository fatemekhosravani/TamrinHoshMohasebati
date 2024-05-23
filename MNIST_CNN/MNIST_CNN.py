import numpy as np
import tensorflow as tf
from tf.keras.layers import Flatten,Dense
from tf.keras.models import Sequential
from tf.keras.optimizers import Adam
from tf.keras.losses import SparseCategoricalCrossentropy
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def normalize_img(image, label):

  return tf.cast(image, tf.float32) / 255., label
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
model = Sequential([
  Flatten(input_shape=(28, 28)),
  Dense(128, activation='relu'),
  Dense(10,activation='relu'),
  Dense(10,activation='softmax')
])
model.compile(
    optimizer=Adam(0.001),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
)
predictions = model.predict(ds_test)
test_images = np.concatenate([x for x, y in ds_test], axis=0)
test_labels = np.concatenate([y for x, y in ds_test], axis=0)
predicted_labels = np.argmax(predictions, axis=1)
image_to_display = test_images[0]
prediction_to_display = predicted_labels[0]
label_to_display = test_labels[0]
plt.figure(figsize=(5, 5))
plt.imshow(image_to_display, cmap='gray')
plt.title(f"Pred: {prediction_to_display}, True: {label_to_display}")
plt.axis('off')
plt.show()
