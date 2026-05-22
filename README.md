# MedTriage-CXR

MedTriage-CXR is an AI-assisted chest X-ray triage system for multi-class classification of RSNA chest X-ray images.

## Project Goal

The goal is to classify chest X-ray images into three triage-relevant classes:

- Normal
- No Lung Opacity / Not Normal
- Lung Opacity

These classes are mapped to triage priorities:

- Normal → Low Priority
- No Lung Opacity / Not Normal → Medium Priority
- Lung Opacity → High Priority

## Project Scope

This project includes:

- DICOM image loading
- Data validation for corrupted or incomplete DICOM files
- Train/validation/test split creation
- PyTorch Dataset and DataLoader pipeline
- ResNet18 baseline model
- DenseNet121 stronger baseline model
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Training curves and experiment reports
- Future Grad-CAM explainability and Streamlit demo

## Disclaimer

This project is for educational and research purposes only. It is not intended for clinical diagnosis or real-world medical use.