# ByteRCNN-Hierarchical: Deep Learning Based File Fragment Type Identification

## Overview

ByteRCNN-Hierarchical is an AI-powered file fragment classification system designed to identify file types using raw binary byte fragments. Instead of relying on file extensions, metadata, or magic-byte signatures, the system learns internal byte patterns through deep learning to predict both the broad file family and the specialized subtype.

This project extends the ByteRCNN research concept by incorporating a hierarchical architecture with one broad classifier and multiple specialist classifiers for more scalable and practical file fragment identification.

## Motivation

In digital forensics, cybersecurity, malware analysis, and damaged file recovery, investigators often encounter incomplete or fragmented files. Traditional identification techniques can fail in those situations. This project addresses that problem by enabling fragment-based file type identification directly from raw byte content.

## Key Features

* Hierarchical classification architecture
* Broad file family identification
* Specialist subtype prediction
* Raw byte fragment analysis
* No dependency on file extensions
* No dependency on metadata signatures
* Interactive Streamlit frontend
* Support for multiple file families and subtypes

## Supported File Families

* Documents
* Images
* Audio
* Video
* Archives
* Source Code
* Executables
* Structured Data

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

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Pillow
* python-docx
* openpyxl
* python-pptx
* reportlab
* FFmpeg

## Data

The `data/` folder contains scripts used to generate 512-byte fragments in CSV format for training and evaluation. The original ByteRCNN work references open-access datasets such as FFT-75 and related fragment classification datasets for audio, video, text, and images.

Relevant references from the original project:

* ByteRCNN paper: [IEEE Access](https://ieeexplore.ieee.org/document/10347203)
* FFT-75 dataset: [IEEE DataPort](https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset)

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important Note

This system is intended for research and educational use. Since fragment-based classification is probabilistic, accuracy may vary for compressed, encrypted, or structurally similar file types.

## Citation

If you use the original ByteRCNN research foundation, please cite:

```bibtex
@ARTICLE{
  10347203,
  author={Skracic, Kristian and Petrovic, Juraj and Pale, Predrag},
  journal={IEEE Access},
  title={ByteRCNN: Enhancing File Fragment Type Identification with Recurrent and Convolutional Neural Networks},
  year={2023},
  pages={1-1},
  doi={10.1109/ACCESS.2023.3340441}
}
```
