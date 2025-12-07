import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

model = tf.keras.models.load_model("../../model/cnn_model.h5")

with open("../../model/class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

st.title("Intel Image Classification - CNN App")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(150,150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    st.write(f"Predicted Class: **{idx_to_class[class_idx]}**")
