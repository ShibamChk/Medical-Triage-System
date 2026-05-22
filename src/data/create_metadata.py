from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    CLASS_INFO_PATH,
    TRAIN_IMAGES_DIR,
    CLASS_TO_LABEL,
    TRIAGE_MAPPING,
)


def main():
    print("Creating classification metadata...")

    if not CLASS_INFO_PATH.exists():
        raise FileNotFoundError(
            f"Class info file not found: {CLASS_INFO_PATH}\n"
            "Make sure the RSNA dataset is extracted into data/raw."
        )

    if not TRAIN_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Train image directory not found: {TRAIN_IMAGES_DIR}\n"
            "Make sure the RSNA dataset is extracted into data/raw."
        )

    class_info = pd.read_csv(CLASS_INFO_PATH)

    required_columns = ["patientId", "class"]

    for column in required_columns:
        if column not in class_info.columns:
            raise ValueError(f"Missing required column: {column}")

    # For classification, we need one row per patient/image.
    metadata = class_info.drop_duplicates(subset=["patientId"]).copy()

    metadata["label"] = metadata["class"].map(CLASS_TO_LABEL)
    metadata["triage_priority"] = metadata["class"].map(TRIAGE_MAPPING)

    if metadata["label"].isna().sum() > 0:
        unknown_classes = metadata[metadata["label"].isna()]["class"].unique()
        raise ValueError(f"Unknown class labels found: {unknown_classes}")

    metadata["label"] = metadata["label"].astype(int)

    # Store relative image paths instead of absolute paths.
    # This makes the project portable across machines and drives.
    metadata["image_relative_path"] = metadata["patientId"].apply(
        lambda patient_id: str(
            Path("data") / "raw" / "stage_2_train_images" / f"{patient_id}.dcm"
        )
    )

    metadata["image_exists"] = metadata["image_relative_path"].apply(
        lambda relative_path: (PROJECT_ROOT / relative_path).exists()
    )

    missing_count = (~metadata["image_exists"]).sum()

    print("Total metadata rows:", len(metadata))
    print("\nClass distribution:")
    print(metadata["class"].value_counts())
    print("\nMissing image files:", missing_count)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "classification_metadata.csv"
    metadata.to_csv(output_path, index=False)

    print(f"\nSaved metadata to: {output_path}")


if __name__ == "__main__":
    main()