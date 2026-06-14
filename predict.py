# import sys
# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras


# # Must match training
# MAXLEN = 4096

# # High-level real-world file categories
# CATEGORY_MAP = {
#     0: "TEXT",
#     1: "WEB",
#     2: "AUDIO",
#     3: "IMAGE",
#     4: "DOCUMENT",
#     5: "DOCUMENT",
#     6: "VIDEO",
#     7: "TEXT",
#     8: "EXECUTABLE",
#     9: "DATA",
#     10: "OTHER"
# }

# def read_fragment(filepath, maxlen=4096):
#     """
#     Reads ONLY raw bytes (file fragment).
#     File extension is ignored.
#     """
#     with open(filepath, "rb") as f:
#         data = f.read(maxlen)

#     if len(data) < maxlen:
#         data += b"\x00" * (maxlen - len(data))

#     return np.frombuffer(data, dtype=np.uint8)

# def main():
#     if len(sys.argv) != 4:
#         print("USAGE:")
#         print("python predict.py <model_path> <scenario> <fragment_file>")
#         sys.exit(1)

#     model_path = sys.argv[1]
#     scenario = sys.argv[2]
#     fragment_path = sys.argv[3]

#     # --------------------------------------------------
#     # Load model
#     # --------------------------------------------------
#     print("Loading trained model...")
#     model = tf.keras.models.load_model(
#         model_path,
#         compile=False,
#         safe_mode=False
#     )

#     # --------------------------------------------------
#     # Load labels
#     # --------------------------------------------------
#     print("Loading labels...")
#     with open("../classes.json", "r") as f:
#         classes = json.load(f)

#     labels = classes[scenario]

#     # --------------------------------------------------
#     # Read fragment
#     # --------------------------------------------------
#     print("Reading file fragment (raw bytes)...")
#     x = read_fragment(fragment_path, MAXLEN)
#     x = np.expand_dims(x, axis=0)

#     # --------------------------------------------------
#     # Predict
#     # --------------------------------------------------
#     print("Predicting file fragment type...")
#     probs = model.predict(x, verbose=0)[0]

#     # Top-K predictions
#     TOP_K = 3
#     top_indices = np.argsort(probs)[-TOP_K:][::-1]

#     # --------------------------------------------------
#     # Output
#     # --------------------------------------------------
#     print("\nFILE FRAGMENT CLASSIFICATION RESULT")
#     print("------------------------------------")

#     for rank, idx in enumerate(top_indices, start=1):
#         numeric_label = int(labels[idx])
#         file_type = CATEGORY_MAP.get(numeric_label, "UNKNOWN")
#         confidence = probs[idx]

#         print(f"{rank}. {file_type:<12}  confidence: {confidence:.4f}")

#     print("\nNote:")
#     print("- Prediction is based ONLY on raw byte fragment")
#     print("- File extension is NOT used")
#     print("- Multiple outputs shown due to fragment ambiguity")

# if __name__ == "__main__":
#     main()





import numpy as np
import tensorflow as tf

# --------------------------------
# SETTINGS
# --------------------------------
MODEL_PATH = r"D:\major\ByteRCNN-main\output_s2_improved\best_model_resume.keras"
FILE_TO_PREDICT = r"D:\major\ByteRCNN-main\sample.zip"

MAXLEN = 1536

# --------------------------------
# LOAD MODEL
# --------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded")

# --------------------------------
# READ FILE AS BYTES
# --------------------------------
with open(FILE_TO_PREDICT, "rb") as f:
    byte_data = f.read()

# convert bytes → integers
x = np.frombuffer(byte_data, dtype=np.uint8)

print("Original length:", len(x))

# --------------------------------
# PAD OR TRUNCATE
# --------------------------------
if len(x) < MAXLEN:
    pad_width = MAXLEN - len(x)
    x = np.pad(x, (0, pad_width))
else:
    x = x[:MAXLEN]

# --------------------------------
# NORMALIZATION
# IMPORTANT → SAME AS TRAINING
# --------------------------------
x = x.astype("int32")

# add batch dimension
x = np.expand_dims(x, axis=0)

print("Input shape:", x.shape)

CLASS_NAMES = {
    0: "archive",
    1: "audio",
    2: "document",
    3: "executable",
    4: "image",
    5: "source_code",
    6: "structured_data",
    7: "text",
    8: "video",
    9: "web",
    10: "other"
}

# --------------------------------
# PREDICTION
# --------------------------------
pred = model.predict(x)

predicted_class = np.argmax(pred)

print("\n✅ Prediction completed")
print("Predicted class:", predicted_class)
print("Predicted type:", CLASS_NAMES[predicted_class])

confidence = np.max(pred)

print("Confidence:", float(confidence))

print("\nClass probabilities:")
print(pred)