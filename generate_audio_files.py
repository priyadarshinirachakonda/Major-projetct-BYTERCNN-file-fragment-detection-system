import os
import random
import subprocess
import numpy as np
from scipy.io.wavfile import write

# =========================================
# OUTPUT BASE
# =========================================

BASE = r"D:\major\ByteRCNN-main\specialists\audio"

# =========================================
# AUDIO TYPES
# =========================================

TYPES = [
    "wav",
    "mp3",
    "flac",
    "ogg",
    "m4a",
    "aiff",
    "wma"
]

# =========================================
# CREATE FOLDERS
# =========================================

for t in TYPES:

    folder = os.path.join(BASE, t)

    os.makedirs(folder, exist_ok=True)

# =========================================
# SETTINGS
# =========================================

COUNT = 200

SAMPLE_RATE = 44100

DURATION = 3

# =========================================
# GENERATE FILES
# =========================================

for i in range(COUNT):

    # ---------------------------------
    # RANDOM AUDIO
    # ---------------------------------

    samples = np.random.uniform(
        -1,
        1,
        SAMPLE_RATE * DURATION
    )

    audio = np.int16(
        samples * 32767
    )

    temp_wav = f"temp_{i}.wav"

    write(
        temp_wav,
        SAMPLE_RATE,
        audio
    )

    # ---------------------------------
    # SAVE WAV
    # ---------------------------------

    wav_path = os.path.join(
        BASE,
        "wav",
        f"{i}.wav"
    )

    os.replace(
        temp_wav,
        wav_path
    )

    # ---------------------------------
    # CONVERSIONS
    # ---------------------------------

    conversions = {

        "mp3": ".mp3",
        "flac": ".flac",
        "ogg": ".ogg",
        "m4a": ".m4a",
        "aiff": ".aiff",
        "wma": ".wma"
    }

    for folder_name, ext in conversions.items():

        output_path = os.path.join(
            BASE,
            folder_name,
            f"{i}{ext}"
        )

        command = [

            "ffmpeg",

            "-y",

            "-i",
            wav_path,

            output_path
        ]

        try:

            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        except Exception as e:

            print(f"❌ ERROR: {folder_name}")

            print(e)

    print(f"✅ GENERATED {i+1}/{COUNT}")

print("\n🔥 AUDIO FILES GENERATED")