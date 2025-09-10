import os, json, math, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ====== إعدادات عامة ======
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_DIR = "train"
VAL_DIR   = "val"
TEST_DIR  = "test"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "egypt_cnn.h5")
CLASS_MAP_PATH = os.path.join(MODELS_DIR, "class_indices.json")

os.makedirs(MODELS_DIR, exist_ok=True)
#generate images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_aug = train_datagen

plain_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=SEED
)
val_gen = plain_aug.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
test_gen = plain_aug.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

num_classes = train_gen.num_classes
print(f"[INFO]num classes {num_classes}")
print("[INFO] class indices:", train_gen.class_indices)
#بيفاح فايل dict
with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump({int(v): k for k, v in train_gen.class_indices.items()}, f, ensure_ascii=False, indent=2)

def build_cnn(input_shape=(224,224,3), num_classes=21):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="EgyptCNN")
    return model

model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
#early stopping
es = callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
ckpt = callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[es, ckpt, rlr]
)


print("\n[INFO] Evaluating on TEST set...")
test_loss, test_acc = model.evaluate(test_gen)
print(f"[RESULT] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# في حالة أفضل وزن اتحفظ في ModelCheckpoint، نتأكد نحمله
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
