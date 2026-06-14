import os

BASE = r"D:\major\ByteRCNN-main\specialists\video"

for folder in os.listdir(BASE):

    path = os.path.join(BASE, folder)

    if os.path.isdir(path):

        count = len(os.listdir(path))

        print(folder, ":", count)