import os
import random
import string
import zipfile

# =========================================
# BASE FOLDER
# =========================================

BASE = r"D:\major\ByteRCNN-main\specialists\docs"

# =========================================
# TYPES
# =========================================

TYPES = [
    "epub",
    "mobi"
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
# RANDOM TEXT GENERATOR
# =========================================

def random_text(length=5000):

    chars = string.ascii_letters + string.digits + " \n"

    return "".join(
        random.choice(chars)
        for _ in range(length)
    )

# =========================================
# GENERATE EPUB FILES
# =========================================

for i in range(COUNT):

    # -------------------------------------
    # EPUB
    # -------------------------------------

    epub_path = os.path.join(
        BASE,
        "epub",
        f"{i}.epub"
    )

    with zipfile.ZipFile(
        epub_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        # mimetype
        z.writestr(
            "mimetype",
            "application/epub+zip"
        )

        # random chapter text
        z.writestr(
            "chapter1.txt",
            random_text(
                random.randint(5000,20000)
            )
        )

        # metadata
        z.writestr(
            "metadata.xml",
            f"""
            <book>
                <title>Book {i}</title>
                <author>AI Generator</author>
            </book>
            """
        )

    # -------------------------------------
    # MOBI
    # -------------------------------------

    mobi_path = os.path.join(
        BASE,
        "mobi",
        f"{i}.mobi"
    )

    with open(mobi_path, "wb") as f:

        # MOBI-like header
        f.write(b"BOOKMOBI")

        # random binary content
        f.write(
            os.urandom(
                random.randint(
                    50000,
                    500000
                )
            )
        )

    print(f"✅ GENERATED {i+1}/{COUNT}")

print("\n🔥 EPUB + MOBI FILES GENERATED")