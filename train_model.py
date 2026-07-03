"""
train_model.py  -  Train the ResNet50 brain tumor classifier.

HOW TO USE
----------
1. Download a dataset from Kaggle:
   - Binary (Yes/No):  https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
   - Multi-class:      https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

2. Organise your images into this folder structure inside the project:
      dataset/
          yes/    <-- MRI scans WITH a tumor
          no/     <-- MRI scans WITHOUT a tumor

3. Run:
      python train_model.py

4. The best model is saved to  model/brain_tumor_model.h5
   Restart app.py and it will automatically use the newly trained weights.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# CONFIG  — change these if needed
# ─────────────────────────────────────────────
DATASET_DIR  = "dataset"          # root folder with yes/ and no/ sub-folders
MODEL_SAVE   = os.path.join("model", "brain_tumor_model.h5")
IMG_SIZE     = 224                # must match app.py
BATCH_SIZE   = 32
EPOCHS_HEAD  = 10                 # phase 1: train only the new head
EPOCHS_FINE  = 10                 # phase 2: fine-tune top ResNet50 layers
LEARNING_RATE = 1e-4
FINE_TUNE_LR  = 1e-5
VAL_SPLIT    = 0.2               # 20% of data used for validation

# ─────────────────────────────────────────────
# STEP 1 — Data generators with augmentation
# ─────────────────────────────────────────────
print("\n[1/4] Setting up data generators...")

# Training augmentation — helps the model generalise better
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # ResNet50-specific normalisation
    validation_split=VAL_SPLIT,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

# Validation — NO augmentation, only preprocessing
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VAL_SPLIT,
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",          # 0 = no tumor, 1 = tumor
    subset="training",
    shuffle=True,
    seed=42,
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=42,
)

print(f"   Training samples : {train_gen.samples}")
print(f"   Validation samples: {val_gen.samples}")
print(f"   Classes           : {train_gen.class_indices}")

# ─────────────────────────────────────────────
# STEP 2 — Build the model
# ─────────────────────────────────────────────
print(f"\n[2/4] Building VGG16 model...")

# Load VGG16 backbone with ImageNet weights, no classification head
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False      # freeze backbone for phase 1

# Custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)   # binary output: 0-1

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
print(f"   Total params    : {model.count_params():,}")
print(f"   Trainable params: {sum(p.numpy().size for p in model.trainable_variables):,}")

os.makedirs("model", exist_ok=True)

# ─────────────────────────────────────────────
# STEP 3 — Phase 1: Train the head only
# ─────────────────────────────────────────────
print(f"\n[3/4] Phase 1 — Training classification head ({EPOCHS_HEAD} epochs)...")

callbacks_phase1 = [
    ModelCheckpoint(
        MODEL_SAVE,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_phase1,
)

# ─────────────────────────────────────────────
# STEP 4 — Phase 2: Fine-tune top ResNet50 layers
# ─────────────────────────────────────────────
print(f"\n[4/4] Phase 2 — Fine-tuning VGG16 block5 conv layers ({EPOCHS_FINE} epochs)...")

# Unfreeze the last conv block of VGG16 (block5: last 4 conv layers)
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with a much lower learning rate
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [
    ModelCheckpoint(
        MODEL_SAVE,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1,
    ),
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_phase2,
)

# ─────────────────────────────────────────────
# RESULTS — Plot training history
# ─────────────────────────────────────────────
print(f"\n✅ Training complete! Best model saved to: {MODEL_SAVE}")

# Combine both phases for plotting
acc     = history1.history["accuracy"]     + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss    = history1.history["loss"]         + history2.history["loss"]
val_loss= history1.history["val_loss"]     + history2.history["val_loss"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Brain Tumor Classifier — Training History (VGG16)", fontsize=14)

# Accuracy plot
axes[0].plot(acc,     label="Train Accuracy",      linewidth=2)
axes[0].plot(val_acc, label="Validation Accuracy",  linewidth=2)
axes[0].axvline(x=EPOCHS_HEAD - 1, color="gray", linestyle="--", label="Fine-tune start")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(loss,     label="Train Loss",      linewidth=2)
axes[1].plot(val_loss, label="Validation Loss",  linewidth=2)
axes[1].axvline(x=EPOCHS_HEAD - 1, color="gray", linestyle="--", label="Fine-tune start")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join("model", "training_history.png")
plt.savefig(plot_path, dpi=150)
print(f"📊 Training plot saved to: {plot_path}")
plt.show()

# Final validation accuracy summary
final_val_acc = max(val_acc)
print(f"\n📈 Best validation accuracy: {final_val_acc * 100:.2f}%")
print("   Restart app.py to use the newly trained model.")
