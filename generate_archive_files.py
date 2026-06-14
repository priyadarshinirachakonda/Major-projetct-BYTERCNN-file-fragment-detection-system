import os
import random

# =========================================
# BASE FOLDER
# =========================================

BASE = r"D:\major\ByteRCNN-main\specialists\archive"

# =========================================
# ARCHIVE TYPES
# =========================================

TYPES = [
    "deb",
    "pkg",
    "rar"
]

# =========================================
# CREATE FOLDERS
# =========================================

for t in TYPES:

    os.makedirs(
        os.path.join(BASE, t),
        exist_ok=True
    )

# =========================================
# SETTINGS
# =========================================

COUNT = 200

# =========================================
# SIMPLE HEADERS
# =========================================

HEADERS = {

    "deb": b"!<arch>\n",

    "pkg": b"xar!",

    "rar": b"Rar!\x1A\x07\x00"
}

# =========================================
# GENERATE FILES
# =========================================

for i in range(COUNT):

    # -------------------------------------
    # RANDOM FILE SIZE
    # -------------------------------------

    size = random.randint(
        50000,
        500000
    )

    # -------------------------------------
    # RANDOM BYTE CONTENT
    # -------------------------------------

    data = bytearray(
        os.urandom(size)
    )

    # -------------------------------------
    # RANDOM STRUCTURE MODIFICATIONS
    # -------------------------------------

    for _ in range(20):

        pos = random.randint(
            0,
            len(data)-1000
        )

        block_size = random.randint(
            100,
            1000
        )

        mode = random.choice([
            "zero",
            "repeat",
            "random"
        ])

        # ---------------------------------
        # ZERO BLOCK
        # ---------------------------------

        if mode == "zero":

            data[
                pos:pos+block_size
            ] = b"\x00" * block_size

        # ---------------------------------
        # REPEATED PATTERN
        # ---------------------------------

        elif mode == "repeat":

            pattern = bytes([
                random.randint(0,255)
            ]) * block_size

            data[
                pos:pos+block_size
            ] = pattern

        # ---------------------------------
        # RANDOM BLOCK
        # ---------------------------------

        elif mode == "random":

            data[
                pos:pos+block_size
            ] = os.urandom(block_size)

    # =====================================
    # SAVE FOR EACH TYPE
    # =====================================

    for ext in TYPES:

        header = HEADERS[ext]

        final_data = bytearray(header)

        final_data.extend(data)

        path = os.path.join(
            BASE,
            ext,
            f"{i}.{ext}"
        )

        with open(path, "wb") as f:

            f.write(final_data)

    print(f"✅ GENERATED {i+1}/{COUNT}")

print("\n🔥 ARCHIVE FILES GENERATED")