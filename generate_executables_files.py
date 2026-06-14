import os
import random
import sqlite3
import struct
import string

# =========================================
# BASE FOLDER
# =========================================

BASE = r"D:\major\ByteRCNN-main\specialists\structured"

# =========================================
# TYPES
# =========================================

TYPES = [
    "db",
    "dwg",
    "pcap",
    "sqlite",
    "ttf"
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

COUNT = 20

# =========================================
# RANDOM TEXT
# =========================================

def random_text(length=20):

    chars = string.ascii_letters

    return "".join(
        random.choice(chars)
        for _ in range(length)
    )

# =========================================
# GENERATE FILES
# =========================================

for i in range(COUNT):

    # =====================================
    # SQLITE FILE
    # =====================================

    sqlite_path = os.path.join(
        BASE,
        "sqlite",
        f"{i}.sqlite"
    )

    conn = sqlite3.connect(sqlite_path)

    cursor = conn.cursor()

    table_name = random_text(8)

    cursor.execute(
        f"""
        CREATE TABLE {table_name} (
            id INTEGER,
            name TEXT,
            value REAL
        )
        """
    )

    for _ in range(50):

        cursor.execute(
            f"""
            INSERT INTO {table_name}
            VALUES (?, ?, ?)
            """,
            (
                random.randint(1,1000),
                random_text(10),
                random.random()*100
            )
        )

    conn.commit()
    conn.close()

    # =====================================
    # DB FILE
    # =====================================

    db_path = os.path.join(
        BASE,
        "db",
        f"{i}.db"
    )

    with open(db_path, "wb") as f:

        f.write(b"DB")

        f.write(
            os.urandom(
                random.randint(
                    50000,
                    200000
                )
            )
        )

    # =====================================
    # DWG FILE
    # =====================================

    dwg_path = os.path.join(
        BASE,
        "dwg",
        f"{i}.dwg"
    )

    with open(dwg_path, "wb") as f:

        # AutoCAD-like signature
        f.write(b"AC1032")

        f.write(
            os.urandom(
                random.randint(
                    100000,
                    500000
                )
            )
        )

    # =====================================
    # PCAP FILE
    # =====================================

    pcap_path = os.path.join(
        BASE,
        "pcap",
        f"{i}.pcap"
    )

    with open(pcap_path, "wb") as f:

        # PCAP Global Header
        f.write(
            struct.pack(
                "IHHIIII",
                0xa1b2c3d4,
                2,
                4,
                0,
                0,
                65535,
                1
            )
        )

        # Fake packets
        for _ in range(100):

            packet_size = random.randint(
                64,
                1500
            )

            # packet header
            f.write(
                struct.pack(
                    "IIII",
                    random.randint(1,999999),
                    0,
                    packet_size,
                    packet_size
                )
            )

            # packet data
            f.write(
                os.urandom(packet_size)
            )

    # =====================================
    # TTF FILE
    # =====================================

    ttf_path = os.path.join(
        BASE,
        "ttf",
        f"{i}.ttf"
    )

    with open(ttf_path, "wb") as f:

        # TTF signature
        f.write(b"\x00\x01\x00\x00")

        f.write(
            os.urandom(
                random.randint(
                    50000,
                    300000
                )
            )
        )

    print(f"✅ GENERATED {i+1}/{COUNT}")

print("\n🔥 STRUCTURED FILES GENERATED")