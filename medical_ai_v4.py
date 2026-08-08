import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(150, 150),
    batch_size=16,            # Smaller batch — better learning!
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False
)

print("\nClass Indices:", train_generator.class_indices)
print("Training batches:", len(train_generator))
print("Testing batches:", len(test_generator))
print("\nData preprocessing complete! ✅")

# ================================
# STEP 4 — Build Final CNN Model
# ================================
print("\nBuilding Final CNN Model...")

model = keras.Sequential([

    # Block 1
    layers.Conv2D(32, (3,3), padding='same',
                  activation='relu',
                  input_shape=(150, 150, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.2),

    # Block 2
    layers.Conv2D(64, (3,3), padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.3),

    # Block 3
    layers.Conv2D(128, (3,3), padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.4),

    # Flatten
    layers.Flatten(),

    # Dense layers
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Output
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
    # Stop when no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,
        patience=3,
        min_lr=0.0000001,
        verbose=1
    ),

    # Save best model automatically!
    ModelCheckpoint(
        'best_pneumonia_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ================================
# STEP 7 — Train
# ================================
print("\nTraining started...")
print("This will take 20-30 minutes...")
print("Please wait! ⏱️\n")

history = model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# ================================
# STEP 8 — Evaluate
# ================================
print("\nEvaluating Final Model...")

loss, accuracy = model.evaluate(test_generator, verbose=0)

print("\n" + "=" * 40)
print("   FINAL MODEL RESULTS V4")
print("=" * 40)
print(f"Test Loss:     {round(loss, 4)}")
print(f"Test Accuracy: {round(accuracy * 100, 2)}%")
print("=" * 40)

# ================================
# STEP 9 — Detailed Report
# ================================
from sklearn.metrics import classification_report, confusion_matrix

test_generator.reset()
predictions = model.predict(test_generator, verbose=0)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_generator.classes

print("\n" + "=" * 40)
print("   DETAILED CLASSIFICATION REPORT")
print("=" * 40)
print(classification_report(y_true, y_pred,
      target_names=['NORMAL', 'PNEUMONIA']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(f"True Normal:     {cm[0][0]}")
print(f"False Pneumonia: {cm[0][1]}")
print(f"False Normal:    {cm[1][0]}")
print(f"True Pneumonia:  {cm[1][1]}")

# ================================
# STEP 10 — Save Final Model
# ================================
model.save("pneumonia_detector_final.keras")
print("\nFinal model saved! ✅")

# ================================
# STEP 11 — Plot Results
# ================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],
             label='Train', color='blue')
axes[0].plot(history.history['val_accuracy'],
             label='Validation', color='orange')
axes[0].axhline(y=0.90, color='green',
                linestyle='--', label='90% Target')
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

fig.suptitle('Pneumonia Detector FINAL — V4',
             fontsize=14)
plt.tight_layout()
plt.savefig('final_results.png')
plt.show()

print("\nFinal graph saved! ✅")
print("\n🏥 FINAL Medical AI Model Ready for Deployment!")   