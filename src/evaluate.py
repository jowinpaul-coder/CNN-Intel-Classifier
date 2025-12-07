import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATASET_PATH = "intel-image-classification"
test_dir = os.path.join(DATASET_PATH, "seg_test/seg_test")

img_height, img_width = 150, 150
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

model = tf.keras.models.load_model("../model/cnn_model.h5")

loss, acc = model.evaluate(test_generator)

print(f"Test Accuracy: {acc * 100:.2f}%")
