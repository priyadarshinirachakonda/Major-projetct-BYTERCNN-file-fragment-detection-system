import os
import random

# =========================================
# SOURCE + OUTPUT
# =========================================

SOURCE = r"D:\major\ByteRCNN-main\specialists\images\ai"

OUTPUT = SOURCE

# =========================================
# SETTINGS
# =========================================

AUGMENT_PER_FILE = 20

# =========================================
# GET FILES
# =========================================

files = [

    f for f in os.listdir(SOURCE)

    if f.lower().endswith(".ai")
]

# =========================================
# AUGMENT
# =========================================

counter = 0

for filename in files:

    path = os.path.join(
        SOURCE,
        filename
    )

    with open(path, "rb") as f:

        data = bytearray(f.read())

    for i in range(AUGMENT_PER_FILE):

        new_data = bytearray(data)

        mode = random.choice([1,2,3])

        # ---------------------------------
        # APPEND RANDOM BYTES
        # ---------------------------------

        if mode == 1:

            extra = os.urandom(
                random.randint(100,5000)
            )

            new_data.extend(extra)

        # ---------------------------------
        # PREPEND RANDOM BYTES
        # ---------------------------------

        elif mode == 2:

            extra = os.urandom(
                random.randint(100,5000)
            )

            new_data = bytearray(extra) + new_data

        # ---------------------------------
        # MODIFY RANDOM REGION
        # ---------------------------------

        elif mode == 3:

            if len(new_data) > 100:

                pos = random.randint(
                    0,
                    len(new_data)-100
                )

                for j in range(50):

                    new_data[pos+j] = random.randint(
                        0,
                        255
                    )

        # ---------------------------------
        # SAVE
        # ---------------------------------

        new_name = f"aug_{counter}.ai"

        save_path = os.path.join(
            OUTPUT,
            new_name
        )

        with open(save_path, "wb") as f:

            f.write(new_data)

        counter += 1

        print(f"✅ {new_name}")

print("\n🔥 AI AUGMENTATION COMPLETE")