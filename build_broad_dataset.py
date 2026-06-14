import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

# =========================================
# SETTINGS
# =========================================

MAXLEN = 1536

NUM_FRAGMENTS = 15

OUTPUT_PATH = r"D:\major\ByteRCNN-main\datasets\broad_custom"

# =========================================
# BROAD LABELS
# =========================================

LABELS = {

    "image": 0,
    "document": 1,
    "audio": 2,
    "video": 3,
    "archive": 4,
    "source_code": 5,
    "executable": 6,
    "structured_data": 7
}

# =========================================
# SOURCE PATHS
# =========================================

FAMILY_PATHS = {

    "image": r"D:\major\ByteRCNN-main\specialists\images",

    "document": r"D:\major\ByteRCNN-main\specialists\docs",

    "audio": r"D:\major\ByteRCNN-main\specialists\audio",

    "video": r"D:\major\ByteRCNN-main\specialists\video",

    "archive": r"D:\major\ByteRCNN-main\specialists\archive",

    "source_code": r"D:\major\ByteRCNN-main\specialists\source_code",

    "structured_data": r"D:\major\ByteRCNN-main\specialists\structured_data",

    "executable": r"D:\major\ByteRCNN-main\specialists\executables"
}

samples = []

# =========================================
# COLLECT FILES
# =========================================

for family_name, label in LABELS.items():

    base_folder = FAMILY_PATHS[family_name]

    print(f"\n📂 Reading family: {family_name}")

    for root, dirs, files in os.walk(base_folder):

        for filename in files:

            file_path = os.path.join(root, filename)

            samples.append((file_path, label))

print(f"\n🔥 TOTAL FILES: {len(samples)}")

# =========================================
# RANDOM FRAGMENT GENERATION
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
# SPLIT
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

print("\n🔥 Generating TRAIN...")
x_train, y_train = generate_fragments(train_files)

print("\n🔥 Generating VAL...")
x_val, y_val = generate_fragments(val_files)

print("\n🔥 Generating TEST...")
x_test, y_test = generate_fragments(test_files)

# =========================================
# SAVE
# =========================================

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

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

print("\n🔥 BROAD DATASET CREATED")
print("Saved to:", OUTPUT_PATH)