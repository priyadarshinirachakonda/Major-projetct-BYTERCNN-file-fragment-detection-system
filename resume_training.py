import tensorflow as tf
import numpy as np

# -----------------------------
# LOAD SAVED MODEL
# -----------------------------
MODEL_PATH = r"D:\major\ByteRCNN-main\output_s2_improved\run_20260508_104809\best_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model restored successfully")

# -----------------------------
# LOAD DATA
# -----------------------------
data = np.load(r"D:\major\4k_2\train.npz")

x_train = data["x"][:60000, :1536]
y_train = data["y"][:60000]

x_train = x_train.astype("int32")

val = np.load(r"D:\major\4k_2\val.npz")

x_val = val["x"][:10000, :1536]
y_val = val["y"][:10000]

x_val = x_val.astype("int32")

# -----------------------------
# TF DATASET
# -----------------------------
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)

train_dataset = train_dataset.shuffle(20000).batch(16).prefetch(
    tf.data.AUTOTUNE
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (x_val, y_val)
)

val_dataset = val_dataset.batch(16).prefetch(
    tf.data.AUTOTUNE
)

# -----------------------------
# CONTINUE TRAINING
# -----------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    verbose=1,
    callbacks=[

        tf.keras.callbacks.ModelCheckpoint(
            r"output_s2_improved\best_model_resume.keras",
            save_best_only=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

print("✅ Resume training completed")