from pathlib import Path


# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_IMAGES_DIR = RAW_DATA_DIR / "stage_2_train_images"
TEST_IMAGES_DIR = RAW_DATA_DIR / "stage_2_test_images"

TRAIN_LABELS_PATH = RAW_DATA_DIR / "stage_2_train_labels.csv"
CLASS_INFO_PATH = RAW_DATA_DIR / "stage_2_detailed_class_info.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR = PROJECT_ROOT / "models"


# -----------------------------
# Class mappings
# -----------------------------
CLASS_TO_LABEL = {
    "Normal": 0,
    "No Lung Opacity / Not Normal": 1,
    "Lung Opacity": 2,
}

LABEL_TO_CLASS = {
    0: "Normal",
    1: "No Lung Opacity / Not Normal",
    2: "Lung Opacity",
}

CLASS_NAMES = [
    "Normal",
    "No Lung Opacity / Not Normal",
    "Lung Opacity",
]

TRIAGE_MAPPING = {
    "Normal": "Low Priority",
    "No Lung Opacity / Not Normal": "Medium Priority",
    "Lung Opacity": "High Priority",
}