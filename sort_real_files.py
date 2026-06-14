import os
import shutil

# =========================================
# SOURCE FOLDER
# =========================================

SOURCE_FOLDER = r"C:\Users\dell\Downloads\7z2601-src"

# =========================================
# DESTINATION ROOT
# =========================================

DEST_ROOT = r"D:\major\ByteRCNN-main\specialists"

# =========================================
# EXTENSION MAP
# =========================================

EXTENSION_MAP = {

    # ---------------------------------
    # DOCUMENTS
    # ---------------------------------

    ".pdf": ("docs", "pdf"),
    ".doc": ("docs", "docx"),
    ".docx": ("docs", "docx"),
    ".ppt": ("docs", "pptx"),
    ".pptx": ("docs", "pptx"),
    ".xls": ("docs", "xlsx"),
    ".xlsx": ("docs", "xlsx"),
    ".txt": ("docs", "txt"),
    ".csv": ("docs", "csv"),

    # ---------------------------------
    # IMAGES
    # ---------------------------------

    ".jpg": ("images", "jpg"),
    ".jpeg": ("images", "jpg"),
    ".png": ("images", "png"),
    ".gif": ("images", "gif"),
    ".bmp": ("images", "bmp"),
    ".tiff": ("images", "tiff"),
    ".tif": ("images", "tiff"),

    # ---------------------------------
    # AUDIO
    # ---------------------------------

    ".mp3": ("audio", "mp3"),
    ".wav": ("audio", "wav"),

    # ---------------------------------
    # VIDEO
    # ---------------------------------

    ".mp4": ("video", "mp4"),
    ".avi": ("video", "avi"),

    # ---------------------------------
    # ARCHIVES
    # ---------------------------------

    ".zip": ("archive", "zip"),
    ".rar": ("archive", "rar"),
    ".7z": ("archive", "7z"),
    ".tar": ("archive", "tar"),
    ".gz": ("archive", "gz")
}

# =========================================
# COUNTERS
# =========================================

moved = 0
skipped = 0

# =========================================
# WALK THROUGH FILES
# =========================================

for root, dirs, files in os.walk(SOURCE_FOLDER):

    for filename in files:

        file_path = os.path.join(root, filename)

        ext = os.path.splitext(filename)[1].lower()

        # ---------------------------------
        # CHECK EXTENSION
        # ---------------------------------

        if ext in EXTENSION_MAP:

            family, subtype = EXTENSION_MAP[ext]

            dest_folder = os.path.join(
                DEST_ROOT,
                family,
                subtype
            )

            os.makedirs(dest_folder, exist_ok=True)

            dest_path = os.path.join(
                dest_folder,
                filename
            )

            # ---------------------------------
            # HANDLE DUPLICATES
            # ---------------------------------

            if os.path.exists(dest_path):

                base, extension = os.path.splitext(filename)

                counter = 1

                while os.path.exists(dest_path):

                    new_name = f"{base}_{counter}{extension}"

                    dest_path = os.path.join(
                        dest_folder,
                        new_name
                    )

                    counter += 1

            # ---------------------------------
            # MOVE FILE
            # ---------------------------------

            try:

                shutil.move(
                    file_path,
                    dest_path
                )

                moved += 1

                print(f"✅ MOVED: {filename}")

            except Exception as e:

                print(f"❌ ERROR: {filename}")

                print(e)

        else:

            unknown_folder = os.path.join(
                DEST_ROOT,
                "unknown"
            )

            os.makedirs(
                unknown_folder,
                exist_ok=True
            )

            dest_path = os.path.join(
                unknown_folder,
                filename
            )

            try:

                shutil.move(
                    file_path,
                    dest_path
                )

                skipped += 1

                print(f"⚠️ UNKNOWN MOVED: {filename}")

            except Exception as e:

                print(f"❌ ERROR: {filename}")

                print(e)

# =========================================
# DONE
# =========================================

print("\n=================================")
print(f"✅ MOVED FILES   : {moved}")
print(f"⚠️ SKIPPED FILES : {skipped}")
print("=================================")