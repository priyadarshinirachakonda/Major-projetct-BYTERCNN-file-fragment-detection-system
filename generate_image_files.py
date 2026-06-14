import os
import random
import numpy as np

from PIL import Image
from PIL import ImageDraw

BASE = r"D:\major\ByteRCNN-main\specialists\images"

TYPES = [
    "jpg",
    "png",
    "gif",
    "bmp",
    "tiff"
]

for t in TYPES:

    os.makedirs(
        os.path.join(BASE, t),
        exist_ok=True
    )

COUNT = 200

for i in range(COUNT):

    arr = np.random.randint(
        0,
        255,
        (256,256,3),
        dtype=np.uint8
    )

    img = Image.fromarray(arr)

    draw = ImageDraw.Draw(img)

    for _ in range(10):

        x1 = random.randint(0,255)
        y1 = random.randint(0,255)

        x2 = random.randint(x1,255)
        y2 = random.randint(y1,255)

        draw.rectangle(
            [x1,y1,x2,y2],
            outline="white"
        )

    for ext in TYPES:

        path = os.path.join(
            BASE,
            ext,
            f"{i}.{ext}"
        )

        img.save(path)

    print(f"✅ GENERATED {i+1}/{COUNT}")

print("\n🔥 IMAGE FILES GENERATED")