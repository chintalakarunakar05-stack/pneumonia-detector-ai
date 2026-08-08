import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ================================
# STEP 1 — Dataset Paths
# ================================
TRAIN_PATH = "chest_xray/train"
TEST_PATH  = "chest_xray/test"

# ================================
# STEP 2 — Check Dataset
# ================================
train_normal    = len(os.listdir(TRAIN_PATH + "/NORMAL"))
train_pneumonia = len(os.listdir(TRAIN_PATH + "/PNEUMONIA"))
test_normal     = len(os.listdir(TEST_PATH + "/NORMAL"))
test_pneumonia  = len(os.listdir(TEST_PATH + "/PNEUMONIA"))

print("=" * 40)
print("   DATASET SUMMARY")
print("=" * 40)
print(f"Train Normal:    {train_normal}")
print(f"Train Pneumonia: {train_pneumonia}")
print(f"Test Normal:     {test_normal}")
print(f"Test Pneumonia:  {test_pneumonia}")
print(f"Total Images:    {train_normal + train_pneumonia + test_normal + test_pneumonia}")
print("=" * 40)

# ================================
# STEP 3 — Image Generators
# ================================
print("\nSetting up Image Generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False
)

print("\nClass Indices:", train_generator.class_indices)
print("Training batches:", len(train_generator))
print("Testing batches:", len(test_generator))
print("\nData preprocessing complete! ✅")

# ================================
# STEP 4 — Build CNN Model
# ================================
print("\nBuilding CNN Model...")

model = keras.Sequential([

    # Block 1
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(150, 150, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    # Block 4
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    # Flatten
    layers.Flatten(),

    # Dense layers
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# ================================
# STEP 5 — Compile
# ================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# STEP 6 — Callbacks
# ================================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )
]

# ================================
# STEP 7 — Train
# ================================
print("\nTraining started...")
print("This will take 15-25 minutes...")
print("Please wait! ⏱️\n")

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# ================================
# STEP 8 — Evaluate
# ================================
loss, accuracy = model.evaluate(test_generator, verbose=0)

print("\n" + "=" * 40)
print("   FINAL MODEL RESULTS")
print("=" * 40)
print(f"Test Loss:     {round(loss, 4)}")
print(f"Test Accuracy: {round(accuracy * 100, 2)}%")
print("=" * 40)

# ================================
# STEP 9 — Save Model
# ================================
model.save("pneumonia_detector_v2.keras")
print("\nModel saved! ✅")

# ================================
# STEP 10 — Plot Results
# ================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],
             label='Train', color='blue')
axes[0].plot(history.history['val_accuracy'],
             label='Validation', color='orange')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],
             label='Train', color='blue')
axes[1].plot(history.history['val_loss'],
             label='Validation', color='orange')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

fig.suptitle('Pneumonia Detector — Session 3 Fixed',
             fontsize=14)
plt.tight_layout()
plt.savefig('session3_results.png')
plt.show()

print("\nGraph saved! ✅")
print("\n🏥 Medical AI Model Ready!")


# Create check.py in same folder
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load saved model
model = keras.models.load_model("pneumonia_detector_v2.keras")

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "chest_xray/test",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False
)

# Evaluate
loss, accuracy = model.evaluate(test_generator, verbose=0)
print("\n" + "=" * 40)
print("   SAVED MODEL RESULTS")
print("=" * 40)
print(f"Test Loss:     {round(loss, 4)}")
print(f"Test Accuracy: {round(accuracy * 100, 2)}%")
print("=" * 40)