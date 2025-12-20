import tensorflow as tf

DATASET_PATH = "intel-image-classification/seg_train/seg_train"

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(150,150),
    batch_size=32
)

model = tf.keras.models.load_model("model/cnn_model.h5")

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}")
