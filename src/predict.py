import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import sys
import os

# =====================
# LOAD MODEL
# =====================
MODEL_PATH = "model/cnn_model.keras"
CLASS_INDEX_PATH = "model/class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# =====================
# PREDICT FUNCTION
# =====================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    print("\nClass probabilities:")
    for i, prob in enumerate(predictions):
        print(f"{idx_to_class[i]}: {prob:.3f}")

    predicted_class = idx_to_class[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"\nFinal prediction: {predicted_class}")
    print(f"Confidence: {confidence:.3f}")


# =====================
# CLI ENTRY
# =====================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print("❌ Image path does not exist")
        sys.exit(1)

    predict_image(img_path)
