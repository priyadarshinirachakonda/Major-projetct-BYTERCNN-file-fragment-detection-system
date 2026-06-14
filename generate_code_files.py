import os
import random

BASE = r"D:\major\ByteRCNN-main\specialists\source_code"

TEMPLATES = {

    "py": "print('hello world')",
    "java": "public class Test {}",
    "cpp": "#include<iostream>",
    "html": "<html><body>Hello</body></html>",
    "css": "body { color:red; }",
    "js": "console.log('hello');",
    "sql": "SELECT * FROM users;",
    "json": '{"name":"test"}',
    "xml": "<root><a>1</a></root>",
    "csv": "id,name\n1,test",
    "txt": "sample text",
    "md": "# Title",
    "log": "[INFO] Server started"
}

COUNT = 200

for ext, content in TEMPLATES.items():

    folder = os.path.join(BASE, ext)

    os.makedirs(folder, exist_ok=True)

    for i in range(COUNT):

        path = os.path.join(
            folder,
            f"{i}.{ext}"
        )

        with open(path, "w", encoding="utf-8") as f:

            for _ in range(random.randint(20,100)):

                f.write(content + "\n")

    print(f"✅ {ext}")

print("🔥 CODE FILES GENERATED")