import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Intel Image Classification using CNN")
st.write("Upload an image and the model will classify it into one of the 6 categories.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/cnn_model.h5")
    return model

model = load_model()

# Class names (Intel dataset)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]

    st.write("### 🏷️ Prediction:", predicted_class)
