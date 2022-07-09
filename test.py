"""This file iterates through the test data and shows what percentage the model predicted correctly."""

import os

import numpy as np
import tensorflow as tf

height = 256
width = 256

classes = ["normal", "phonecall_left", "phonecall_right", "phonehold_left", "phonehold_right"]

class_to_test = classes[0]
model_path = "modelex.h5"  # EDIT THIS

model = tf.keras.models.load_model(model_path)

# load images
images = os.listdir(f"data/test/{class_to_test}/")
correct = 0
for image in images:
    # print(image)
    img = tf.keras.utils.load_img(
        f"data/test/{class_to_test}/{image}", target_size=(height, width),
        # grayscale=True
    )
    img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # print(f"Class: {classes[np.argmax(score)]} - Confidence: {100 * np.max(score)}")

    if classes[np.argmax(score)] == class_to_test:
        correct += 1
print(f"Correct: {correct}/{len(images)} ({100 * correct / len(images)}%)")