import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Loading best transfer model...")
model = keras.models.load_model('best_transfer_model.keras')
print("Model loaded! ✅")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "chest_xray/test",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator, verbose=1)

print("\n" + "=" * 45)
print("   BEST TRANSFER MODEL RESULTS")
print("=" * 45)
print(f"Test Loss:     {round(loss, 4)}")
print(f"Test Accuracy: {round(accuracy * 100, 2)}%")
print("=" * 45)
print(f"\n📊 Previous CNN    → 88.62%")
print(f"📊 Transfer Model  → {round(accuracy * 100, 2)}%")
print(f"📊 Improvement     → +{round((accuracy*100)-88.62, 2)}%")