import os
import random

# =========================================
# PSD FOLDER
# =========================================

PSD_FOLDER = r"D:\major\ByteRCNN-main\specialists\images\psd"

# =========================================
# SETTINGS
# =========================================

AUGMENT_PER_FILE = 25

# =========================================
# GET PSD FILES
# =========================================

files = [

    f for f in os.listdir(PSD_FOLDER)

    if f.lower().endswith(".psd")
]

# =========================================
# COUNTER
# =========================================

counter = 0

# =========================================
# AUGMENT
# =========================================

for filename in files:

    path = os.path.join(
        PSD_FOLDER,
        filename
    )

    with open(path, "rb") as f:

        original = bytearray(
            f.read()
        )

    for i in range(AUGMENT_PER_FILE):

        data = bytearray(original)

        mode = random.choice([
            "append",
            "prepend",
            "modify",
            "mix"
        ])

        # ---------------------------------
        # APPEND RANDOM BYTES
        # ---------------------------------

        if mode == "append":

            extra = os.urandom(
                random.randint(100,10000)
            )

            data.extend(extra)

        # ---------------------------------
        # PREPEND RANDOM BYTES
        # ---------------------------------

        elif mode == "prepend":

            extra = os.urandom(
                random.randint(100,5000)
            )

            data = bytearray(extra) + data

        # ---------------------------------
        # MODIFY RANDOM REGION
        # ---------------------------------

        elif mode == "modify":

            if len(data) > 1000:

                pos = random.randint(
                    0,
                    len(data)-500
                )

                for j in range(200):

                    data[pos+j] = random.randint(
                        0,
                        255
                    )

        # ---------------------------------
        # MIX RANDOM BLOCKS
        # ---------------------------------

        elif mode == "mix":

            if len(data) > 5000:

                start1 = random.randint(
                    0,
                    len(data)-2000
                )

                start2 = random.randint(
                    0,
                    len(data)-2000
                )

                block1 = data[
                    start1:start1+1000
                ]

                block2 = data[
                    start2:start2+1000
                ]

                data[
                    start1:start1+1000
                ] = block2

                data[
                    start2:start2+1000
                ] = block1

        # ---------------------------------
        # SAVE
        # ---------------------------------

        save_name = f"aug_{counter}.psd"

        save_path = os.path.join(
            PSD_FOLDER,
            save_name
        )

        with open(save_path, "wb") as f:

            f.write(data)

        counter += 1

        print(f"✅ {save_name}")

print("\n🔥 PSD AUGMENTATION COMPLETE")