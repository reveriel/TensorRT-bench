
# ref https://www.tensorflow.org/tutorials/images/transfer_learning
# ResNet, R

from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow_datasets as tfds
import os

import numpy as np

# import matplotlib.pyplot as plt

import tensorflow as tf

keras = tf.keras

# tfds.disable_progress_bar()

SPLIT_WEIGHTS = (8, 1, 1)
# splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

# (raw_train, raw_validation, raw_test), metadata = tfds.load(
#     'cats_vs_dogs', split=list(splits),
#     with_info=True, as_supervised=True)

# get_label_name = metadata.features['label'].int2str

# for image, label in raw_train.take(2):
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))

IMG_SIZE = 224  # All images will be resized to 160x160


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


# train = raw_train.map(format_example)
# validation = raw_validation.map(format_example)
# test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

# train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# validation_batches = validation.batch(BATCH_SIZE)
# test_batches = test.batch(BATCH_SIZE)


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model Resnet V2
# see https://keras.io/applications/#resnet
base_model = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE,
                                                     include_top=False,
                                                     weights='imagenet', classes=1000)

base_model = tf.keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3),
                                                     include_top=False,
                                                     weights='imagenet', classes=1000)
# feature_batch = base_model(image_batch)

base_model.trainable = False

for i in range(1000):
    output = base_model(np.random.rand(2, 224, 224, 3))
    print(output.shape)


# base_learning_rate = 0.0001
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


# initial_epochs = 10
# steps_per_epoch = round(num_train)//BATCH_SIZE
# validation_steps = 20

# loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
