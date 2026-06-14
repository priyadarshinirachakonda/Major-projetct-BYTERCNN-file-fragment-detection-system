import os
import numpy as np
from sklearn.model_selection import train_test_split
import random

# =========================================
# SETTINGS
# =========================================

DATASET_PATH = r"D:\major\ByteRCNN-main\specialists\archive"

OUTPUT_PATH = r"D:\major\ByteRCNN-main\datasets\archive"

MAXLEN = 1024

NUM_FRAGMENTS = 5

# =========================================
# LABELS
# =========================================

LABELS = {
    "zip": 0,
    "gz": 1,
    "bz2": 2,
    "xz": 3,
    "tar": 4,
    "7z": 5,
    "deb": 6,
    "pkg": 7,
    "rar": 8,
    "rpm": 9
}

# =========================================
# STORAGE
# =========================================

samples = []

# =========================================
# GENERATE FRAGMENTS
# =========================================

def generate_fragments(file_list):

    X = []
    Y = []

    for file_path, label in file_list:

        try:

            with open(file_path, "rb") as f:

                byte_data = f.read()

            x = np.frombuffer(
                byte_data,
                dtype=np.uint8
            )

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

                    fragment = x[
                        offset : offset + MAXLEN
                    ]

                X.append(fragment)
                Y.append(label)

        except Exception as e:

            print("\n❌ ERROR:", file_path)
            print(e)

    X = np.array(X, dtype=np.uint8)

    Y = np.array(Y, dtype=np.int32)

    return X, Y

# =========================================
# READ FILES
# =========================================

for class_name, label in LABELS.items():

    folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    print(f"\n📂 Reading: {class_name}")

    if not os.path.exists(folder):

        print("❌ Missing folder:", folder)

        continue

    files = os.listdir(folder)

    print(f"Found {len(files)} files")

    for filename in files:

        file_path = os.path.join(
            folder,
            filename
        )

        samples.append((file_path, label))

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

# =========================================
# GENERATE DATASETS
# =========================================

print("\n🔥 Generating TRAIN fragments...")
x_train, y_train = generate_fragments(train_files)

print("\n🔥 Generating VAL fragments...")
x_val, y_val = generate_fragments(val_files)

print("\n🔥 Generating TEST fragments...")
x_test, y_test = generate_fragments(test_files)

print("\n===================================")
print("TRAIN:", x_train.shape)
print("VAL:", x_val.shape)
print("TEST:", x_test.shape)

# =========================================
# CREATE OUTPUT FOLDER
# =========================================

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

# =========================================
# SAVE DATASETS
# =========================================

np.savez_compressed(
    os.path.join(
        OUTPUT_PATH,
        "train.npz"
    ),
    x=x_train,
    y=y_train
)

np.savez_compressed(
    os.path.join(
        OUTPUT_PATH,
        "val.npz"
    ),
    x=x_val,
    y=y_val
)

np.savez_compressed(
    os.path.join(
        OUTPUT_PATH,
        "test.npz"
    ),
    x=x_test,
    y=y_test
)

print("\n===================================")
print("🔥 ARCHIVE SPECIALIST DATASET CREATED")
print("Saved to:", OUTPUT_PATH)