from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR


def check_patient_overlap(train_df, val_df, test_df):
    train_patients = set(train_df["patientId"])
    val_patients = set(val_df["patientId"])
    test_patients = set(test_df["patientId"])

    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)

    print("Train-Val overlap:", len(train_val_overlap))
    print("Train-Test overlap:", len(train_test_overlap))
    print("Val-Test overlap:", len(val_test_overlap))

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError("Patient leakage detected between splits.")


def show_distribution(df: pd.DataFrame, split_name: str):
    print(f"\n{split_name} distribution:")
    print(df["class"].value_counts())
    print("\nPercentage:")
    print((df["class"].value_counts(normalize=True) * 100).round(2))


def main():
    metadata_path = PROCESSED_DATA_DIR / "classification_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Run: python src/data/create_metadata.py"
        )

    metadata = pd.read_csv(metadata_path)

    required_columns = [
        "patientId",
        "class",
        "label",
        "triage_priority",
        "image_relative_path",
        "image_exists",
    ]

    for column in required_columns:
        if column not in metadata.columns:
            raise ValueError(f"Missing required column: {column}")

    # Remove rows whose image file is missing.
    metadata = metadata[metadata["image_exists"] == True].copy()

    print("Rows after removing missing images:", len(metadata))

    train_df, temp_df = train_test_split(
        metadata,
        test_size=0.30,
        random_state=42,
        stratify=metadata["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    check_patient_overlap(train_df, val_df, test_df)

    show_distribution(train_df, "Train")
    show_distribution(val_df, "Validation")
    show_distribution(test_df, "Test")

    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved splits:")
    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()