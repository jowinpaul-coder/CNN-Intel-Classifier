import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

model = tf.keras.models.load_model("../model/cnn_model.h5")

with open("../model/class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)

    print(f"Predicted class: {idx_to_class[class_idx]}")

predict_image("sample.jpg")
