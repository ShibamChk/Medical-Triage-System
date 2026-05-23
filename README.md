# MedTriage-CXR

MedTriage-CXR is an end-to-end AI-assisted chest X-ray triage system built for multi-class medical image classification, sensitivity-aware decision tuning, explainability, and deployable inference.

The system classifies chest X-ray images into three triage-relevant categories:

- Normal
- No Lung Opacity / Not Normal
- Lung Opacity

These classes are mapped into triage priorities:

- Normal → Low Priority
- No Lung Opacity / Not Normal → Medium Priority
- Lung Opacity → High Priority

> Disclaimer: This project is for educational and portfolio purposes only. It is not intended for clinical diagnosis or real-world medical decision making.

---

## Project Motivation

In medical triage, accuracy alone is not enough. A model can achieve reasonable overall accuracy while still missing high-priority cases.

This project focuses on building a full machine learning pipeline that includes:

- DICOM data processing
- corrupted file validation
- train/validation/test splitting
- modern CNN-based classification
- high-sensitivity threshold tuning
- Grad-CAM explainability
- Streamlit demo app
- FastAPI inference endpoint

The main goal is not only to classify X-rays, but also to demonstrate how an ML system can be designed for practical triage workflows.

---

## Dataset

This project uses the RSNA Pneumonia Detection Challenge dataset.

The dataset contains chest X-ray DICOM images and metadata labels. For this project, the original labels are converted into a three-class classification task:

| Class | Label | Triage Priority |
|---|---:|---|
| Normal | 0 | Low Priority |
| No Lung Opacity / Not Normal | 1 | Medium Priority |
| Lung Opacity | 2 | High Priority |

The raw dataset is not included in this repository because of size and licensing restrictions.

Expected local data structure:

```text
data/
├── raw/
│   ├── stage_2_train_images/
│   ├── stage_2_test_images/
│   ├── stage_2_train_labels.csv
│   └── stage_2_detailed_class_info.csv
└── processed/
    ├── classification_metadata.csv
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── clean_train.csv
    ├── clean_val.csv
    └── clean_test.csv

---

## System Pipeline

The complete system follows an end-to-end machine learning workflow:

```text
Raw RSNA DICOM Dataset
        ↓
Metadata Creation
        ↓
Train/Validation/Test Split
        ↓
DICOM File Validation
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Threshold Tuning for Triage Sensitivity
        ↓
Grad-CAM Explainability
        ↓
Streamlit Demo + FastAPI Inference API

---

## Visual Results

### ConvNeXt-Tiny Confusion Matrix
![ConvNeXt Confusion Matrix](reports/figures/confusion_matrix_convnext_tiny_safe.png)

### Threshold Tuning Curve
![Threshold Curve](reports/figures/threshold_curve_convnext_tiny_threshold_tuned.png)

### High-Sensitivity Triage Confusion Matrix
![Tuned Confusion Matrix](reports/figures/confusion_matrix_tuned_normalized_convnext_tiny_threshold_tuned.png)

### Grad-CAM Explainability Gallery
![Grad-CAM Gallery](reports/figures/convnext_tiny_gradcam_gallery.png)

---

## Demo Interfaces

### Streamlit Demo

The Streamlit app allows users to upload a chest X-ray image, run inference, choose between balanced and high-sensitivity triage mode, view class probabilities, and generate Grad-CAM explanations.

![Streamlit Demo 1](reports/figures/streamlit_demo_1.png)

![Streamlit Demo 2](reports/figures/streamlit_demo_2.png)

### FastAPI Inference API

The FastAPI endpoint provides production-style model serving through REST API endpoints. It supports health checks, model information, and file-based prediction through Swagger UI.

![FastAPI Swagger Docs](reports/figures/fastapi_docs.png)

---

medtriage-cxr/
├── app/
│   ├── streamlit_app.py
│   └── api.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/
│   └── README.md
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── create_metadata.py
│   │   ├── create_splits.py
│   │   ├── dataset.py
│   │   └── validate_dicom_files.py
│   ├── models/
│   │   └── model.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── plot_training_curves.py
│   │   └── tune_thresholds.py
│   ├── explainability/
│   │   ├── gradcam.py
│   │   └── gradcam_gallery.py
│   └── inference/
│       └── predict.py
├── tests/
├── requirements.txt
├── README.md
└── main.py

---

Author

Shibam Chakraborty