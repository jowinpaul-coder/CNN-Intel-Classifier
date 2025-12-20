import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.title("Intel Image Classification (MobileNetV2)")

model = tf.keras.models.load_model("model/cnn_model.keras")

with open("model/class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224,224))
    st.image(img, use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)[0]
    pred_class = idx_to_class[np.argmax(preds)]

    st.success(f"Prediction: {pred_class}")
    st.write(f"Confidence: {np.max(preds):.2f}")
