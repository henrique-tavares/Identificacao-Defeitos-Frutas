import csv
from os import path
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

with open(path.join(path.curdir, "databases", "pequis.csv")) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    csv_rows = [row for row in csv_reader]

    images = [path.join(path.curdir, "pequis", csv_row[0]) for csv_row in csv_rows[1:]]
    labels = [csv_row[1] for csv_row in csv_rows[1:]]

images = io.imread_collection(images)

train_images, test_images = (images[: int(len(images) * 0.8)], images[int(len(images) * 0.8) :])
train_labels, test_labels = (labels[: int(len(labels) * 0.8)], labels[int(len(labels) * 0.8) :])

print(train_images[0].shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(512, 512, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(3))

# model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    np.array([image for image in images]),
    labels,
    epochs=5,
    verbose=2,
    validation_split=0.2,
)
