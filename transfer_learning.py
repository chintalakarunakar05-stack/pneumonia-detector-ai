import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
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

print("=" * 45)
print("   TRANSFER LEARNING — PNEUMONIA DETECTOR")
print("=" * 45)
print(f"Train Normal:    {train_normal}")
print(f"Train Pneumonia: {train_pneumonia}")
print(f"Test Normal:     {test_normal}")
print(f"Test Pneumonia:  {test_pneumonia}")
print("=" * 45)

# ================================
# STEP 3 — Image Generators
# ================================
print("\nSetting up Image Generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 150x150 — Faster training!
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(150, 150),   # ✅ Updated!
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(150, 150),   # ✅ Updated!
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

print(f"Class Indices: {train_generator.class_indices}")
print(f"Training batches: {len(train_generator)}")
print(f"Testing batches: {len(test_generator)}")
print("\nGenerators ready! ✅")

# ================================
# STEP 4 — Load VGG16
# ================================
print("\nLoading VGG16 pretrained model...")
print("VGG16 trained on 14 million images!")

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)  # ✅ Updated!
)

# Freeze base model
base_model.trainable = False

print(f"VGG16 layers: {len(base_model.layers)}")
print("Base model frozen! ✅")

# ================================
# STEP 5 — Build Model
# ================================
print("\nBuilding Transfer Learning Model...")

model = keras.Sequential([

    # VGG16 base — pretrained!
    base_model,

    # Our custom layers!
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    # Output
    layers.Dense(1, activation='sigmoid')
])

# ================================
# STEP 6 — Compile
# ================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# STEP 7 — Callbacks
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
    ),
    ModelCheckpoint(
        'best_transfer_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ================================
# STEP 8 — Phase 1 Training
# ================================
print("\n" + "=" * 45)
print("   PHASE 1 — Training Top Layers")
print("=" * 45)
print("VGG16 frozen — training our layers!")
print("Please wait... ⏱️\n")

history1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# ================================
# STEP 9 — Phase 2 Fine Tuning
# ================================
print("\n" + "=" * 45)
print("   PHASE 2 — Fine Tuning VGG16")
print("=" * 45)

# Unfreeze last 4 VGG16 layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Last 4 VGG16 layers unfrozen!")
print("Fine tuning starts... ⏱️\n")

history2 = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# ================================
# STEP 10 — Evaluate
# ================================
loss, accuracy = model.evaluate(test_generator, verbose=0)

print("\n" + "=" * 45)
print("   TRANSFER LEARNING RESULTS")
print("=" * 45)
print(f"Test Loss:     {round(loss, 4)}")
print(f"Test Accuracy: {round(accuracy * 100, 2)}%")
print("=" * 45)

print("\n📊 IMPROVEMENT COMPARISON")
print("=" * 45)
print(f"Previous CNN    → 88.62%")
print(f"Transfer Learning → {round(accuracy * 100, 2)}%")
improvement = round((accuracy * 100) - 88.62, 2)
print(f"Improvement     → +{improvement}%")
print("=" * 45)

# ================================
# STEP 11 — Save Model
# ================================
model.save("pneumonia_detector_VGG16.keras")
print("\nVGG16 model saved! ✅")

# ================================
# STEP 12 — Plot Results
# ================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Combine both phase histories
acc = history1.history['accuracy'] + \
      history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + \
          history2.history['val_accuracy']
loss_h = history1.history['loss'] + \
         history2.history['loss']
val_loss = history1.history['val_loss'] + \
           history2.history['val_loss']

# Accuracy plot
axes[0].plot(acc, label='Train', color='blue')
axes[0].plot(val_acc, label='Validation', color='orange')
axes[0].axhline(y=0.90, color='green',
                linestyle='--', label='90% Target')
axes[0].axhline(y=0.95, color='red',
                linestyle='--', label='95% Target')
axes[0].axvline(
    x=len(history1.history['accuracy'])-1,
    color='purple', linestyle='--',
    label='Fine Tuning Start')
axes[0].set_title('Transfer Learning Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss plot
axes[1].plot(loss_h, label='Train', color='blue')
axes[1].plot(val_loss, label='Validation', color='orange')
axes[1].set_title('Transfer Learning Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

fig.suptitle(
    'VGG16 Transfer Learning — Pneumonia Detector',
    fontsize=14)
plt.tight_layout()
plt.savefig('transfer_learning_results.png')
plt.show()

print("\nGraph saved! ✅")
print("\n🏥 VGG16 Transfer Learning Complete!")
print(f"🎯 Final Accuracy: {round(accuracy * 100, 2)}%")