import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from bytercnn_models import byte_rcnn_model


gpus = tf.config.experimental.list_physical_devices('GPU')

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# -----------------------------
# CONFIG (OPTIMIZED FOR CPU)
# -----------------------------
maxlen = 1024
embed_dim = 32
BATCH_SIZE = 64
kernels = [3, 5, 7]
CNN_SIZE = 128
RNN_SIZE = 64


LR = 0.0003

EPOCHS = 15

OUTPUT = r"D:\major\ByteRCNN-main\models\archive"

# -----------------------------
# CREATE UNIQUE RUN FOLDER
# -----------------------------
import datetime

run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT = os.path.join(OUTPUT, f"run_{run_name}")

os.makedirs(OUTPUT, exist_ok=True)

# -----------------------------
# LOAD TRAIN + VAL
# -----------------------------
print("\n" + "="*80)

DATASET_PATH = r"D:\major\ByteRCNN-main\datasets\archive"

train_data = np.load(
    os.path.join(DATASET_PATH, "train.npz")
)

val_data = np.load(
    os.path.join(DATASET_PATH, "val.npz")
)

test_data = np.load(
    os.path.join(DATASET_PATH, "test.npz")
)

x_train = train_data["x"]
y_train = train_data["y"]

x_val = val_data["x"]
y_val = val_data["y"]

x_test = test_data["x"]
y_test = test_data["y"]

# IMPORTANT
x_train = x_train.astype("int32")
x_val = x_val.astype("int32")
x_test = x_test.astype("int32")

from sklearn.utils import shuffle
x_train, y_train = shuffle(
    x_train,
    y_train,
    random_state=42
)

print(f"TRAIN: {x_train.shape} | VAL: {x_val.shape}")

# -----------------------------
# LABEL NORMALIZATION
# -----------------------------
y_train = y_train
y_val = y_val

num_classes = len(np.unique(y_train))

CLASS_NAMES = [
    "zip",
    "gz",
    "bz2",
    "xz",
    "tar",
    "7z",
    "deb",
    "pkg",
    "rar",
    "rpm"
]

print(f"LABELS AFTER GROUPING: {num_classes}")

# -----------------------------
# MODEL
# -----------------------------
model = byte_rcnn_model(
    maxlen,
    embed_dim,
    RNN_SIZE,
    CNN_SIZE,
    kernels,
    num_classes,
    OUTPUT,
    LR
)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("Class distribution:", np.bincount(y_train))


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    verbose=1,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT, "best_model.keras"),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,   # stronger drop
            patience=1,   # faster reaction
            min_lr=1e-5
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
)

# -----------------------------
# PLOTS
# -----------------------------
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.savefig(os.path.join(OUTPUT, 'accuracy.png'))
plt.clf()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.savefig(os.path.join(OUTPUT, 'loss.png'))
plt.clf()

# -----------------------------
# TEST
# -----------------------------

# apply same label mapping
y_test = y_test

results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("\nTEST LOSS, ACC:", results)

# predictions
y_pred = model.predict(x_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred, axis=-1)

# -----------------------------
# METRICS
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred, average='macro'))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(os.path.join(OUTPUT, "archive_specialist.keras"))
print("\n✅ TRAINING COMPLETED SUCCESSFULLY")


import json

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred, average='macro')),
    "precision": float(precision_score(y_test, y_pred, average='macro')),
    "recall": float(recall_score(y_test, y_pred, average='macro')),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "maxlen": maxlen,
    "train_samples": int(len(x_train))
}

with open(os.path.join(OUTPUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

