# ByteRCNN-Hierarchical: Deep Learning Based File Fragment Type Identification

## Overview

ByteRCNN-Hierarchical is an AI-powered file fragment classification system designed to identify the type of a file using only its raw binary byte fragments. Unlike traditional file identification approaches that depend on file extensions, metadata, or magic-byte signatures, this system analyzes internal byte patterns through deep learning models to determine both the broad file family and the specialized file subtype.

The project extends the ByteRCNN research concept by incorporating a hierarchical architecture consisting of a broad classifier and multiple specialist classifiers, resulting in improved scalability and practical applicability.

---

## Motivation

In domains such as digital forensics, cybersecurity, malware analysis, and damaged file recovery, analysts frequently encounter incomplete, fragmented, or intentionally manipulated files. Conventional identification techniques become ineffective in such scenarios.

This project addresses these challenges by enabling fragment-based file type identification directly from raw byte content.

---

## Key Features

* Hierarchical classification architecture.
* Broad file family identification.
* Specialist subtype prediction.
* Raw byte fragment analysis.
* No dependency on file extensions.
* No dependency on metadata signatures.
* Interactive Streamlit frontend.
* GPU-accelerated deep learning inference.
* Support for multiple file families and subtypes.

---

## Supported File Families

* Documents
* Images
* Audio
* Video
* Archives
* Source Code
* Executables
* Structured Data

---

## System Architecture

Input File

↓

Random Byte Fragment Extraction

↓

Broad Classifier

↓

Specialist Routing

↓

Subtype Classification

↓

Final Prediction

---

## Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### GPU Acceleration

* TensorFlow GPU
* CUDA
* cuDNN

### Data Processing

* NumPy
* Scikit-learn

### Frontend

* Streamlit

### Visualization

* Matplotlib

### File Generation Libraries

* Pillow
* python-docx
* openpyxl
* python-pptx
* reportlab
* FFmpeg

---

## Dataset Characteristics

* Custom fragment-based dataset.
* Combination of real and synthetic files.
* Approximately 300,000+ generated fragments.
* Multiple fragments extracted from each file.
* Fragment sizes ranging from 1024 to 1536 bytes.
* Stratified train-validation-test splitting.

---

## Applications

### Digital Forensics

Identification of fragmented or partially recovered files during forensic investigations.

### Cybersecurity

Detection and categorization of unknown files encountered in security monitoring environments.

### Malware Analysis

Assisting analysts in identifying suspicious file structures.

### Data Recovery

Supporting recovery processes when only file fragments are available.

### Storage System Analysis

Classification of residual file fragments in storage devices.

---

## Project Contributions

Compared to existing research works, this project extends file fragment classification by introducing:

* Hierarchical broad-to-specialist routing.
* Specialist models for fine-grained classification.
* Interactive frontend deployment.
* Practical end-to-end implementation.
* Expanded file family coverage.
* Automated dataset generation pipelines.

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Application

```bash
streamlit run app.py
```

---

## Important Note

This system performs prediction using raw byte fragments and is intended for research and educational purposes. Due to the probabilistic nature of fragment analysis, classification accuracy may vary for highly compressed or structurally similar file types.

---

## Future Scope

* Incremental learning for new file types.
* Improved handling of compressed container formats.
* Explainable AI integration.
* Cloud deployment support.
* Enhanced specialist architectures.
* Real-time forensic integration.

---

## Author

Developed as part of a Major Project focusing on the intersection of Artificial Intelligence, Digital Forensics, and Cybersecurity.
