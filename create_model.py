"""
create_model.py  -  Builds the model ARCHITECTURE and saves it as an .h5 file.

Run this ONCE if you don't yet have a trained model:
    python create_model.py

IMPORTANT: This creates a model with the correct STRUCTURE but UNTRAINED weights.
It lets the web app run and lets you test the full pipeline (upload -> result -> history),
but its predictions will be RANDOM until you replace it with a real trained model.

Backbone: VGG16 (pretrained on ImageNet, top layers frozen)
To get REAL predictions, see README.md -> "Getting a trained model".
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import os

IMG_SIZE = 224

# VGG16 backbone (pretrained on ImageNet, no top), frozen
base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

# Custom classification head
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)   # single neuron, 0-1, binary classification

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

os.makedirs("model", exist_ok=True)
model.save("model/brain_tumor_model.h5")
print("Saved model/brain_tumor_model.h5 — VGG16 backbone (architecture only, untrained).")
