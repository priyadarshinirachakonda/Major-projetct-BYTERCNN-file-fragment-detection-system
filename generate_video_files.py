import os
import cv2
import random
import numpy as np
import subprocess

# =========================================
# BASE
# =========================================

BASE = r"D:\major\ByteRCNN-main\specialists\video"

# =========================================
# LOW COUNT TYPES
# =========================================

TARGET_TYPES = {

    "3gp": 250,
    "flv": 250,
    "mkv": 250,
    "mov": 250,
    "ogv": 250,
    "webm": 250
}

# =========================================
# VIDEO SETTINGS
# =========================================

WIDTH = 320
HEIGHT = 240

FPS = 20

FRAMES = 80

# =========================================
# GENERATE
# =========================================

for video_type, target_count in TARGET_TYPES.items():

    folder = os.path.join(BASE, video_type)

    os.makedirs(folder, exist_ok=True)

    current_count = len(os.listdir(folder))

    needed = target_count - current_count

    print(f"\n🎬 {video_type}")

    print(f"Current : {current_count}")

    print(f"Need    : {needed}")

    if needed <= 0:

        continue

    for i in range(needed):

        # ---------------------------------
        # TEMP MP4
        # ---------------------------------

        temp_mp4 = f"temp_{video_type}_{i}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(

            temp_mp4,
            fourcc,
            FPS,
            (WIDTH, HEIGHT)
        )

        # ---------------------------------
        # CREATE STRUCTURED FRAMES
        # ---------------------------------

        for frame_num in range(FRAMES):

            frame = np.zeros(

                (HEIGHT, WIDTH, 3),

                dtype=np.uint8
            )

            # RANDOM BACKGROUND
            color = (

                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255)
            )

            frame[:] = color

            # MOVING TEXT
            cv2.putText(

                frame,

                f"{video_type} {frame_num}",

                (
                    random.randint(20,100),
                    random.randint(50,200)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (
                    255-color[0],
                    255-color[1],
                    255-color[2]
                ),

                2
            )

            # RANDOM CIRCLES
            for _ in range(5):

                cv2.circle(

                    frame,

                    (
                        random.randint(0, WIDTH),
                        random.randint(0, HEIGHT)
                    ),

                    random.randint(10,50),

                    (
                        random.randint(0,255),
                        random.randint(0,255),
                        random.randint(0,255)
                    ),

                    -1
                )

            out.write(frame)

        out.release()

        # ---------------------------------
        # OUTPUT PATH
        # ---------------------------------

        output_path = os.path.join(

            folder,

            f"generated_{current_count+i}.{video_type}"
        )

        # ---------------------------------
        # FFMPEG CONVERSION
        # ---------------------------------

        command = [

            "ffmpeg",

            "-y",

            "-i",
            temp_mp4,

            output_path
        ]

        try:

            subprocess.run(

                command,

                stdout=subprocess.DEVNULL,

                stderr=subprocess.DEVNULL
            )

        except Exception as e:

            print(f"❌ ERROR: {video_type}")

            print(e)

        # ---------------------------------
        # REMOVE TEMP FILE
        # ---------------------------------

        if os.path.exists(temp_mp4):

            os.remove(temp_mp4)

        print(

            f"✅ {video_type}: "
            f"{i+1}/{needed}"
        )

print("\n🔥 VIDEO GENERATION COMPLETE")