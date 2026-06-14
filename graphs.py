import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =====================================================
# MODELS
# =====================================================

models = [
    "Broad",
    "Source",
    "Audio",
    "Video",
    "Image",
    "Docs",
    "Structured",
    "Archive",
    "Executable"
]

# =====================================================
# OVERALL METRICS
# =====================================================

accuracy = [
    77.20,
    100.0,
    78.65,
    69.27,
    63.04,
    57.88,
    50.91,
    36.29,
    33.33
]

precision = [
    75.84,
    100.0,
    83.79,
    69.46,
    59.50,
    58.92,
    40.88,
    30.62,
    17.32
]

recall = [
    76.39,
    100.0,
    78.00,
    68.65,
    58.69,
    52.86,
    50.00,
    26.31,
    29.75
]

f1 = [
    72.75,
    100.0,
    74.35,
    68.81,
    56.26,
    47.61,
    40.45,
    23.01,
    20.08
]

# =====================================================
# 1️⃣ OVERALL METRICS GRAPH
# =====================================================

x = np.arange(len(models))

width = 0.2

plt.figure(figsize=(18,8))

plt.bar(x - 1.5*width, accuracy, width, label='Accuracy')

plt.bar(x - 0.5*width, precision, width, label='Precision')

plt.bar(x + 0.5*width, recall, width, label='Recall')

plt.bar(x + 1.5*width, f1, width, label='F1 Score')

plt.xticks(x, models, rotation=15)

plt.ylabel("Percentage")

plt.title("Overall Specialist Performance Comparison")

plt.legend()

plt.tight_layout()

plt.savefig("overall_metrics_graph.png")

plt.show()

# =====================================================
# 2️⃣ ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(12,6))

plt.bar(models, accuracy)

plt.ylabel("Accuracy")

plt.title("Accuracy Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("accuracy_graph.png")

plt.show()

# =====================================================
# 3️⃣ PRECISION GRAPH
# =====================================================

plt.figure(figsize=(12,6))

plt.bar(models, precision)

plt.ylabel("Precision")

plt.title("Precision Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("precision_graph.png")

plt.show()

# =====================================================
# 4️⃣ RECALL GRAPH
# =====================================================

plt.figure(figsize=(12,6))

plt.bar(models, recall)

plt.ylabel("Recall")

plt.title("Recall Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("recall_graph.png")

plt.show()

# =====================================================
# 5️⃣ F1 SCORE GRAPH
# =====================================================

plt.figure(figsize=(12,6))

plt.bar(models, f1)

plt.ylabel("F1 Score")

plt.title("F1 Score Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("f1_graph.png")

plt.show()

# =====================================================
# 6️⃣ CLASSIFICATION REPORT HEATMAP
# =====================================================

df = pd.DataFrame({

    "Accuracy": accuracy,

    "Precision": precision,

    "Recall": recall,

    "F1 Score": f1

}, index=models)

plt.figure(figsize=(10,8))

sns.heatmap(

    df,

    annot=True,

    cmap="YlGnBu"

)

plt.title("Overall Classification Report Heatmap")

plt.tight_layout()

plt.savefig("classification_report_heatmap.png")

plt.show()

# =====================================================
# 7️⃣ CONFUSION-LIKE MATRIX
# =====================================================

matrix_data = np.array([
    accuracy,
    precision,
    recall,
    f1
])

plt.figure(figsize=(12,6))

sns.heatmap(

    matrix_data,

    annot=True,

    cmap="Blues",

    xticklabels=models,

    yticklabels=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

)

plt.title("Overall Performance Matrix")

plt.tight_layout()

plt.savefig("overall_confusion_style_matrix.png")

plt.show()


# =====================================================
# PLOT CONFUSION-STYLE MATRIX
# =====================================================

plt.figure(figsize=(14,6))

sns.heatmap(

    matrix_data,

    annot=True,

    fmt=".2f",

    cmap="Blues",

    xticklabels=models,

    yticklabels=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

)

plt.title("Overall Specialists Performance Matrix")

plt.tight_layout()

plt.savefig("overall_confusion_matrix.png")

plt.show()

# =====================================================
# DONE
# =====================================================

print("\n✅ OVERALL CONFUSION MATRIX GENERATED")
# =====================================================
# DONE
# =====================================================

print("\n✅ ALL OVERALL PERFORMANCE GRAPHS GENERATED")