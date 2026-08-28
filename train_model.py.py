import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "garbage_classification"
MODEL_FILE = "waste_classifier.keras"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# ============================================================
# LOAD DATASET
# ============================================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("\nClasses found:")
print(train_generator.class_indices)

# ============================================================
# BUILD CNN MODEL
# ============================================================

model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(train_generator.num_classes, activation="softmax")
])

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n========================================")
print("       WASTE CLASSIFICATION SYSTEM")
print("========================================")
print("\nTraining started...\n")

# ============================================================
# TRAIN
# ============================================================

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_FILE)

print("\n========================================")
print("Model trained successfully!")
print("Saved as:", MODEL_FILE)
print("========================================")

# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names = list(train_generator.class_indices.keys())

with open("classes.txt", "w") as file:
    for class_name in class_names:
        file.write(class_name + "\n")

print("Class names saved to classes.txt")
