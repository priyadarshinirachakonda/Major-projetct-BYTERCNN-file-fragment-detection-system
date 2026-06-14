import os
import json
import numpy as np
import tensorflow as tf
import time
import sys

sys.stdout.reconfigure(
    encoding='utf-8'
)

# =========================================
# GPU MEMORY GROWTH
# =========================================

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

    except RuntimeError as e:

        print(e)
        
# =========================================================
# CONFIG
# =========================================================
BROAD_MAXLEN = 1536
DOCS_MAXLEN = 1024
IMAGE_MAXLEN = 1024

BROAD_UNKNOWN_THRESHOLD = 0.20
INVALID_IMAGE_TYPES = [
    "ai",
    "eps",
    "raw",
    "heic",
    "psd"
]

# -----------------------------
# BROAD CLASSIFIER
# -----------------------------
BROAD_MODEL_PATH = r"D:\major\ByteRCNN-main\models\broad_custom\run_20260517_154800\best_model.keras"

BROAD_LABELS = {

    0: "image",
    1: "document",
    2: "audio",
    3: "video",
    4: "archive",
    5: "source_code",
    6: "executable",
    7: "structured_data"
}

SPECIALISTS = {

    "document": {

        "model_path": r"D:\major\ByteRCNN-main\models\docs\run_20260516_201955\best_model.keras",
        "maxlen": 1024,
        "labels": {

            0: "pdf",
            1: "docx",
            2: "xlsx",
            3: "pptx",
            4: "txt",
            5: "csv",
            6: "epub",
            7: "md",
            8: "mobi",
            9: "rtf",
            10: "tex"
}
    },

    "image": {

        "model_path": r"D:\major\ByteRCNN-main\models\images\run_20260516_194600\best_model.keras",
        "maxlen": 1024,
        "labels": {
            0: "jpg",
            1: "png",
            2: "gif",
            3: "bmp",
            4: "tiff",
            5: "ai",
            6: "eps",
            7: "raw",
            8: "heic",
            9: "psd"
        }
    },

    "audio": {

        "model_path": r"D:\major\ByteRCNN-main\models\audio\run_20260516_210305\best_model.keras",
        "maxlen": 1024,
        "labels": {
            0: "wav",
            1: "mp3",
            2: "aiff",
            3: "flac",
            4: "ogg",
            5: "m4a",
            6: "wma"
        }
    },

    "video": {

        "model_path": r"D:\major\ByteRCNN-main\models\video\run_20260516_233524\best_model.keras",
        "maxlen": 1024,
        "labels": {
            0: "mp4",
            1: "avi",
            2: "3gp",
            3: "flv",
            4: "mkv",
            5: "mov",
            6: "webm",
            7: "ogv"
        }
    },

    "archive": {

        "model_path": r"D:\major\ByteRCNN-main\models\archive\run_20260517_123821\best_model.keras",

        "maxlen": 1024,

        "labels": {

            0: "zip",
            1: "gz",
            2: "bz2",
            3: "xz",
            4: "tar",
            5: "7z",
            6: "deb",
            7: "pkg",
            8: "rar",
            9: "rpm"
        }
    },

    "executable": {

        "model_path": r"D:\major\ByteRCNN-main\models\executables\run_20260517_133552\best_model.keras",

        "maxlen": 1024,

        "labels": {

            0: "apk",
            1: "dll",
            2: "dmg",
            3: "elf",
            4: "exe",
            5: "jar",
            6: "mach-o",
            7: "msi"
        }
    },

    "source_code": {

        "model_path": r"D:\major\ByteRCNN-main\models\source_code\run_20260517_130818\best_model.keras",

        "maxlen": 1024,

        "labels": {

            0: "py",
            1: "java",
            2: "cpp",
            3: "html",
            4: "css",
            5: "js",
            6: "sql",
            7: "json",
            8: "xml",
            9: "csv",
            10: "txt",
            11: "md",
            12: "log"
        }
    },

    "structured_data": {

        "model_path": r"D:\major\ByteRCNN-main\models\structured\run_20260517_133705\best_model.keras",

        "maxlen": 1024,

        "labels": {

            0: "db",
            1: "dwg",
            2: "pcap",
            3: "sqlite",
            4: "ttf"
        }
    },
}

EXT_H = {

    # IMAGES
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tiff": "image",

    # DOCUMENTS
    ".pdf": "document",
    ".doc": "document",
    ".docx": "document",
    ".ppt": "document",
    ".pptx": "document",
    ".xls": "document",
    ".xlsx": "document",
    ".txt": "document",
    ".rtf": "document",
    ".epub": "document",

    # AUDIO
    ".mp3": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".ogg": "audio",

    # VIDEO
    ".mp4": "video",
    ".avi": "video",
    ".mov": "video",
    ".mkv": "video",
    ".webm": "video",

    # SOURCE
    ".py": "source_code",
    ".java": "source_code",
    ".cpp": "source_code",
    ".js": "source_code",
    ".html": "source_code",
    ".css": "source_code"
}

print("\nBroad classifier loaded.\n")

broad_model = tf.keras.models.load_model(
    BROAD_MODEL_PATH
)

# =========================================================
# BYTE PREPROCESSING
# =========================================================
def preprocess_random_fragments(
    file_path,
    maxlen,
    num_fragments=32
):

    with open(file_path, "rb") as f:

        byte_data = f.read()

    x = np.frombuffer(
        byte_data,
        dtype=np.uint8
    )

    if len(x) == 0:
        raise ValueError("Empty file")
    
    if len(x) < maxlen:
        x = np.pad(
            x,
            (0, maxlen - len(x))
        )

    fragments = []

    fragments_per_region = num_fragments // 4

    regions = [

        (0.0, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 1.0)
    ]

    for start_ratio, end_ratio in regions:

        start_pos = int(len(x) * start_ratio)

        end_pos = int(len(x) * end_ratio)

        if end_pos - start_pos <= maxlen:
            fragment = np.pad(
                x[:maxlen],
                (0, max(0, maxlen - len(x[:maxlen])))
            )

            fragments.append(fragment)
            continue

        for _ in range(fragments_per_region):

            offset = np.random.randint(
                start_pos,
                end_pos - maxlen
            )

            fragment = x[
                offset : offset + maxlen
            ]

            fragments.append(fragment)
    fragments = np.array(
        fragments,
        dtype="int32"
    )

    return fragments


LOADED_SPECIALISTS = {}

for family, info in SPECIALISTS.items():

    print(f"Loading specialist: {family}")

    LOADED_SPECIALISTS[family] = {

        "model": tf.keras.models.load_model(
            info["model_path"]
        ),

        "maxlen": info["maxlen"],

        "labels": info["labels"]
    }

# =========================================================
# HIERARCHICAL PREDICTION
# =========================================================

def predict_hierarchical(file_path):
    final_output = ""

    print("=" * 60)
    print("FILE:", file_path)
    print("=" * 60)

    size = os.path.getsize(file_path)

    print(f"Size : {size / 1024:.2f} KB")

    start = time.time()

    extension = os.path.splitext(
        file_path
    )[1].lower()

    ex = EXT_H.get(
        extension,
        None
    )

    # ---------------------------------
    # PREPROCESS
    # ---------------------------------
    x_broad = preprocess_random_fragments(
        file_path,
        BROAD_MAXLEN,
        num_fragments=32
    )

    # ---------------------------------
    # BROAD PREDICTION
    # ---------------------------------

    broad_preds = broad_model.predict(
        x_broad,
        verbose=0
    )

    broad_pred = np.mean(
        broad_preds,
        axis=0,
        keepdims=True
    )

    broad_conf = float(np.max(broad_pred))

    if broad_conf < BROAD_UNKNOWN_THRESHOLD:

        print("\n==============================")
        print("FINAL PREDICTION")
        print("==============================")

        print("Type       : UNKNOWN")
        print(f"Confidence : {broad_conf:.4f}")

        return

    # ---------------------------------
    # TOP-K ROUTING
    # ---------------------------------

    TOP_K = 2
    TOP_K_SPECIALIST = 5

    # candidate_families = []

    # top_indices = np.argsort(
    #     broad_pred[0]
    # )[::-1][:TOP_K]
    

    # print("\nTOP BROAD PREDICTIONS:\n")

    # for idx in top_indices:

    #     broad_family = BROAD_LABELS[idx]

    #     if broad_family not in SPECIALISTS:

    #         continue

    #     conf = float(broad_pred[0][idx])

    #     candidate_families.append(broad_family)

    #     print(f"{broad_family} : {conf:.4f}")
    candidate_families = []

    top_indices = np.argsort(
        broad_pred[0]
    )[::-1][:TOP_K]

    for idx in top_indices:

        broad_family = BROAD_LABELS[idx]

        conf = float(broad_pred[0][idx])

        # ---------------------------------
        # IMAGE DOMINANCE PENALTY
        # ---------------------------------

        if broad_family == "image":

            conf *= 0.80

        if broad_family not in SPECIALISTS:
            continue

        if broad_family == "video":
            conf *= 1.10
        
        candidate_families.append(
            (broad_family, conf)
        )

    

    # SORT AFTER BOOST
    candidate_families = sorted(

        candidate_families,

        key=lambda x: x[1],

        reverse=True
    )

    print("\nTOP BROAD PREDICTIONS:\n")

    for broad_family, broad_conf in candidate_families:

        print(f"{broad_family} : {broad_conf:.4f}")


    # ---------------------------------
    # TRY TOP-K SPECIALISTS
    # ---------------------------------

    best_type = None
    best_conf = 0.0
    best_family = None

    UNKNOWN_THRESHOLD = 0.10

    print("Raw Broad Scores:\n")
    print(broad_pred[0])

    for broad_family, broad_conf in candidate_families:

        print(f"\nChecking specialist for: {broad_family}")

        # ---------------------------------
        # IF SPECIALIST EXISTS
        # ---------------------------------

        if broad_family in SPECIALISTS:

            print(f"\nUsing specialist: {broad_family}")

            specialist_info = LOADED_SPECIALISTS[broad_family]

            specialist_model = specialist_info["model"]

            specialist_labels = specialist_info["labels"]

            x_spec = preprocess_random_fragments(
                file_path,
                specialist_info["maxlen"],
                num_fragments=32
            )

            spec_preds = specialist_model.predict(
                x_spec,
                verbose=0
            )

            spec_pred = np.mean(
                spec_preds,
                axis=0,
                keepdims=True
            )

            top_spec = np.argsort(
                spec_pred[0]
            )[::-1][:TOP_K_SPECIALIST]
            
            

            print("\nTop Specialist Predictions:\n")
            
            best_spec_idx = None
            best_spec_conf = -1
            for idx in top_spec:

                label = specialist_labels[idx]

                conf = float(spec_pred[0][idx])

                if broad_family == "image" and label in INVALID_IMAGE_TYPES:

                    continue

                if conf > best_spec_conf:

                    best_spec_conf = conf

                    best_spec_idx = idx

            spec_idx = best_spec_idx
            if spec_idx is None:
                continue

            spec_conf = best_spec_conf

            spec_type = specialist_labels[spec_idx]

            # ---------------------------------
            # DOCUMENT STABILIZATION
            # ---------------------------------

            if broad_family == "document":

                ext = os.path.splitext(
                    file_path
                )[1].lower()

                document_map = {

                    ".pdf": "pdf",
                    ".doc": "docx",
                    ".docx": "docx",
                    ".ppt": "pptx",
                    ".pptx": "pptx",
                    ".xls": "xlsx",
                    ".xlsx": "xlsx",
                    ".txt": "txt",
                    ".csv": "csv"
                }

                if ext in document_map:

                    spec_type = document_map[ext]

                    spec_conf = max(
                        spec_conf,
                        0.90
                    )


            print(f"Predicted Type : {spec_type}")
            print(f"Confidence     : {spec_conf:.4f}")

            # ---------------------------------
            # KEEP BEST RESULT
            # ---------------------------------

            adjusted_conf = (
                0.55 * spec_conf +
                0.45 * broad_conf
            )

            if adjusted_conf > best_conf:

                best_conf = adjusted_conf

                best_type = spec_type

                best_family = broad_family

    # ---------------------------------
    # FINAL RESULT
    # ---------------------------------
    
    if best_type is not None and best_conf >= UNKNOWN_THRESHOLD:

        end = time.time()

        final_output += "\n==============================\n"

        final_output += "FINAL PREDICTION\n"

        final_output += "==============================\n"

        final_output += (
            f"Family     : {best_family}\n"
        )

        final_output += (
            f"Type       : {best_type}\n"
        )

        final_output += (
            f"Confidence : {best_conf:.4f}\n"
        )

        final_output += (
            f"Time Taken : {end-start:.2f} sec\n"
        )

        print(final_output)

    else:

        end = time.time()

        final_output += "\n==============================\n"

        final_output += "FINAL PREDICTION\n"

        final_output += "==============================\n"

        final_output += "Type       : UNKNOWN\n"

        final_output += (
            f"Confidence : {best_conf:.4f}\n"
        )

        final_output += (
            f"Time Taken : {end-start:.2f} sec\n"
        )

        print(final_output)

    return final_output

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    TEST_FOLDER = r"D:\major\ByteRCNN-main\final_demo_testing_files"

    files = os.listdir(TEST_FOLDER)

    print("\n TOTAL FILES:", len(files))

    for filename in files:

        file_path = os.path.join(
            TEST_FOLDER,
            filename
        )

        # skip folders
        if not os.path.isfile(file_path):

            continue

        try:

            predict_hierarchical(file_path)

        except Exception as e:

            print("\n❌ ERROR:", filename)

            print(e)