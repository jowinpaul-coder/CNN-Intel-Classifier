import tensorflow as tf
from tensorflow.keras import layers, models
import json
import os

# =====================
# CONFIG (FIXED PATHS)
# =====================
TRAIN_DIR = "intel-image-classification/seg_train/seg_train"
VAL_DIR   = "intel-image-classification/seg_test"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

# =====================
# LOAD DATA
# =====================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("✅ Detected classes:", class_names)
print("✅ Number of classes:", num_classes)

# =====================
# PERFORMANCE
# =====================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# =====================
# DATA AUGMENTATION
# =====================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =====================
# MODEL
# =====================
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),

    layers.Rescaling(1./255),
    data_augmentation,

    layers.Conv2D(32, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================
# TRAIN
# =====================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =====================
# SAVE MODEL + LABELS
# =====================
os.makedirs("model", exist_ok=True)

model.save("model/cnn_model.keras")

class_indices = {name: idx for idx, name in enumerate(class_names)}
with open("model/class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=4)

print("✅ Training completed successfully")
print("✅ Model saved to model/cnn_model.keras")
print("✅ Class indices saved")
