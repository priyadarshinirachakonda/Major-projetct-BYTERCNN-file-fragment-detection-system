# import os
# import json
# import sys
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras

# # Fix for keras version mismatch
# import keras
# keras.saving.serialization_lib._retrieve_class_or_fn = tf.keras.saving.serialization_lib._retrieve_class_or_fn
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     accuracy_score,
#     f1_score,
#     precision_score,
#     recall_score
# )
# import matplotlib.pyplot as plt
# from bytercnn_models import byte_rcnn_model

# print("sys.argv =", sys.argv)

# if len(sys.argv) < 5:
#     print("python bytercnn_training_runner.py <scenario> <lr> <epochs> <output_dir>")
#     sys.exit(1)

# SCENARIO_TO_RUN = int(sys.argv[1])
# LR = float(sys.argv[2])
# EPOCHS = int(sys.argv[3])
# OUTPUT = sys.argv[4]

# # --------------------------------------------------
# # CONFIG
# # --------------------------------------------------
# maxlen = 4096
# embed_dim = 16
# BATCH_SIZE = 64     # 🔥 reduced for CPU
# kernels = [9, 27]
# CNN_SIZE = 32
# RNN_SIZE = 32

# if not os.path.exists(OUTPUT):
#     os.makedirs(OUTPUT)

# # --------------------------------------------------
# # HELPERS
# # --------------------------------------------------
# def random_sample_data(x, y, sample_size):
#     idx = np.random.choice(len(x), size=sample_size, replace=False)
#     return x[idx], y[idx]

# def load(scenario=1, block_size="4k", subset="train"):
#     data_dir = os.path.join("..", f"{block_size}_{scenario}")
#     data = np.load(os.path.join(data_dir, f"{subset}.npz"))

#     with open("../classes.json") as f:
#         classes = json.load(f)
#         labels = classes[str(scenario)]

#     return data["x"], data["y"], labels

# # --------------------------------------------------
# # LOAD DATA
# # --------------------------------------------------
# x_train, y_train, labels = load(SCENARIO_TO_RUN, "4k", "train")
# X_val, y_val, _ = load(SCENARIO_TO_RUN, "4k", "val")

# # --------------------------------------------------
# # 🔥 RANDOMIZED DATA SELECTION
# # --------------------------------------------------
# TRAIN_SAMPLES = 10000
# VAL_SAMPLES = 2000

# x_train, y_train = random_sample_data(x_train, y_train, TRAIN_SAMPLES)
# X_val, y_val = random_sample_data(X_val, y_val, VAL_SAMPLES)

# print(f"TRAIN: {len(x_train)} | VAL: {len(X_val)}")

# # --------------------------------------------------
# # MODEL
# # --------------------------------------------------
# model = byte_rcnn_model(
#     maxlen,
#     embed_dim,
#     RNN_SIZE,
#     CNN_SIZE,
#     kernels,
#     len(labels),
#     OUTPUT,
#     LR
# )

# # --------------------------------------------------
# # TRAIN
# # --------------------------------------------------
# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(X_val, y_val),
#     verbose=1,
#     callbacks=[
#         keras.callbacks.ModelCheckpoint(
#             OUTPUT + "best_model.keras",
#             save_best_only=True
#         )
#     ]
# )

# # --------------------------------------------------
# # TEST
# # --------------------------------------------------
# x_test, y_test, labels_val = load(SCENARIO_TO_RUN, "4k", "test")
# results = model.evaluate(x_test[:5000], y_test[:5000], batch_size=BATCH_SIZE)
# print("TEST LOSS, ACC:", results)

# # --------------------------------------------------
# # SAVE MODEL
# # --------------------------------------------------
# model.save(OUTPUT + f"bytercnn_len{maxlen}_sc{SCENARIO_TO_RUN}.keras")

# print("\n✅ TRAINING COMPLETED SUCCESSFULLY")




import os
import json
import sys
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
# ARGUMENTS
# -----------------------------
if len(sys.argv) < 5:
    print("Usage: python bytercnn_trainer.py <scenario> <lr> <epochs> <output_dir>")
    sys.exit(1)

SCENARIO_TO_RUN = int(sys.argv[1])
LR = float(sys.argv[2])
EPOCHS = int(sys.argv[3])
OUTPUT = sys.argv[4]

# -----------------------------
# CREATE UNIQUE RUN FOLDER
# -----------------------------
import datetime

run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT = os.path.join(OUTPUT, f"run_{run_name}")

os.makedirs(OUTPUT, exist_ok=True)

# -----------------------------
# CONFIG (OPTIMIZED FOR CPU)
# -----------------------------
maxlen = 1536
embed_dim = 32
BATCH_SIZE = 16
kernels = [3, 5, 7, 9, 15]
CNN_SIZE = 192
RNN_SIZE = 96

TRAIN_SAMPLES = 60000
VAL_SAMPLES = 10000
TEST_SAMPLES = 10000




# -----------------------------
# LOAD DATA (MEMORY SAFE)
# -----------------------------
def load_sampled(scenario, block_size, subset, sample_size):
    data_dir = r"D:\major\4k_2"
    file_path = os.path.join(data_dir, f"{subset}.npz")

    data = np.load(file_path, mmap_mode='r')

    # total_samples = len(data["y"])

    # # instead of random index
    # np.random.seed(42)
    
    x = data["x"][:sample_size]
    y = data["y"][:sample_size]


    # IMPORTANT → match model input
    x = x[:, :maxlen]   # SAME as maxlen

    x = x.astype("int32")

    return x, y

# -----------------------------
# LOAD TRAIN + VAL
# -----------------------------
print("\n" + "="*80)

x_train, y_train = load_sampled(SCENARIO_TO_RUN, "4k", "train", TRAIN_SAMPLES)
x_val, y_val     = load_sampled(SCENARIO_TO_RUN, "4k", "val", VAL_SAMPLES)

from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train)

print(f"TRAIN: {x_train.shape} | VAL: {x_val.shape}")

# -----------------------------
# LABEL NORMALIZATION
# -----------------------------
y_train = y_train
y_val = y_val

num_classes = len(np.unique(y_train))

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
train_dataset = train_dataset.shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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
x_test, y_test   = load_sampled(SCENARIO_TO_RUN, "4k", "test", TEST_SAMPLES)


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
print("F1:", f1_score(y_test, y_pred, average='micro'))
print("Precision:", precision_score(y_test, y_pred, average='micro'))
print("Recall:", recall_score(y_test, y_pred, average='micro'))

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(os.path.join(OUTPUT, f"bytercnn_sc{SCENARIO_TO_RUN}.keras"))

print("\n✅ TRAINING COMPLETED SUCCESSFULLY")


import json

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred, average='micro')),
    "precision": float(precision_score(y_test, y_pred, average='micro')),
    "recall": float(recall_score(y_test, y_pred, average='micro')),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "maxlen": maxlen,
    "train_samples": TRAIN_SAMPLES
}

with open(os.path.join(OUTPUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

