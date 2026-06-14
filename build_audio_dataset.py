import os
import numpy as np
from sklearn.model_selection import train_test_split
import random

# =========================================
# SETTINGS
# =========================================

DATASET_PATH = r"D:\major\ByteRCNN-main\specialists\audio"

OUTPUT_PATH = r"D:\major\ByteRCNN-main\datasets\audio"

MAXLEN = 1024

# =========================================
# LABELS
# =========================================

LABELS = {
    "wav": 0,
    "mp3": 1,
    "aiff": 2,
    "flac": 3,
    "ogg": 4,
    "m4a": 5,
    "wma": 6
}
# =========================================
# STORAGE
# =========================================
samples = []

def generate_fragments(file_list):

    X = []
    Y = []

    NUM_FRAGMENTS = 5

    for file_path, label in file_list:

        try:

            with open(file_path, "rb") as f:

                byte_data = f.read()

            x = np.frombuffer(byte_data, dtype=np.uint8)

            for _ in range(NUM_FRAGMENTS):

                if len(x) < MAXLEN:

                    fragment = np.pad(
                        x,
                        (0, MAXLEN - len(x))
                    )

                else:

                    offset = random.randint(
                        0,
                        len(x) - MAXLEN
                    )

                    fragment = x[offset:offset + MAXLEN]

                X.append(fragment)
                Y.append(label)

        except Exception as e:

            print("ERROR:", file_path)
            print(e)

    X = np.array(X, dtype=np.uint8)
    Y = np.array(Y, dtype=np.int32)

    return X, Y


# =========================================
# READ FILES
# =========================================

for class_name, label in LABELS.items():

    folder = os.path.join(DATASET_PATH, class_name)

    print(f"\n📂 Reading: {class_name}")

    if not os.path.exists(folder):

        print("❌ Missing folder:", folder)
        continue

    files = os.listdir(folder)

    print(f"Found {len(files)} files")

    for filename in files:

        file_path = os.path.join(folder, filename)

        try:
            samples.append((file_path, label))

        except Exception as e:

            print("\n❌ Error reading:", file_path)
            print(e)


# =========================================
# TRAIN / VAL / TEST SPLIT
# =========================================
train_files, temp_files = train_test_split(
    samples,
    test_size=0.2,
    random_state=42,
    stratify=[label for _, label in samples]
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=0.5,
    random_state=42,
    stratify=[label for _, label in temp_files]
)

print("\n===================================")
print("TRAIN FILES:", len(train_files))
print("VAL FILES:", len(val_files))
print("TEST FILES:", len(test_files))

x_train, y_train = generate_fragments(train_files)
x_val, y_val = generate_fragments(val_files)
x_test, y_test = generate_fragments(test_files)

print("\n===================================")
print("TRAIN:", x_train.shape)
print("VAL:", x_val.shape)
print("TEST:", x_test.shape)

# =========================================
# CREATE OUTPUT FOLDER
# =========================================

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================================
# SAVE DATASETS
# =========================================

np.savez_compressed(
    os.path.join(OUTPUT_PATH, "train.npz"),
    x=x_train,
    y=y_train
)

np.savez_compressed(
    os.path.join(OUTPUT_PATH, "val.npz"),
    x=x_val,
    y=y_val
)

np.savez_compressed(
    os.path.join(OUTPUT_PATH, "test.npz"),
    x=x_test,
    y=y_test
)

print("\n===================================")
print("🔥 Specialist dataset saved")
print("Location:", OUTPUT_PATH)