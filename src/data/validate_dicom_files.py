from pathlib import Path
import sys

import pandas as pd
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, ExplicitVRLittleEndian
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR


SPLIT_FILES = {
    "train": PROCESSED_DATA_DIR / "train.csv",
    "val": PROCESSED_DATA_DIR / "val.csv",
    "test": PROCESSED_DATA_DIR / "test.csv",
}


def has_pixel_data(dicom: pydicom.dataset.FileDataset) -> bool:
    return (
        "PixelData" in dicom
        or "FloatPixelData" in dicom
        or "DoubleFloatPixelData" in dicom
    )


def can_read_dicom_pixels(image_path: str | Path) -> tuple[bool, str]:
    image_path = Path(image_path)

    if not image_path.exists():
        return False, "File does not exist"

    try:
        dicom = pydicom.dcmread(str(image_path), force=True)
    except Exception as error:
        return False, f"Cannot read DICOM file: {repr(error)}"

    if not has_pixel_data(dicom):
        return False, "No Pixel Data found"

    if not hasattr(dicom, "file_meta") or dicom.file_meta is None:
        dicom.file_meta = FileMetaDataset()

    transfer_syntaxes_to_try = [
        None,
        ImplicitVRLittleEndian,
        ExplicitVRLittleEndian,
    ]

    last_error = ""

    for transfer_syntax in transfer_syntaxes_to_try:
        try:
            if transfer_syntax is not None:
                dicom.file_meta.TransferSyntaxUID = transfer_syntax

            if hasattr(dicom, "_pixel_array"):
                delattr(dicom, "_pixel_array")

            _ = dicom.pixel_array

            return True, ""

        except Exception as error:
            last_error = repr(error)

    return False, last_error


def validate_split(split_name: str, csv_path: Path) -> pd.DataFrame:
    print(f"\nValidating {split_name} split")
    print(f"CSV path: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    valid_rows = []
    bad_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Checking {split_name}"):
        image_path = PROJECT_ROOT / row["image_relative_path"]

        is_valid, error_message = can_read_dicom_pixels(image_path)

        if is_valid:
            valid_rows.append(row)
        else:
            bad_row = row.copy()
            bad_row["split"] = split_name
            bad_row["error_message"] = error_message
            bad_rows.append(bad_row)

    clean_df = pd.DataFrame(valid_rows)
    bad_df = pd.DataFrame(bad_rows)

    clean_path = PROCESSED_DATA_DIR / f"clean_{split_name}.csv"
    clean_df.to_csv(clean_path, index=False)

    print(f"{split_name} original rows: {len(df)}")
    print(f"{split_name} clean rows:    {len(clean_df)}")
    print(f"{split_name} bad rows:      {len(bad_df)}")
    print(f"Saved clean split to: {clean_path}")

    return bad_df


def main():
    all_bad_dfs = []

    for split_name, csv_path in SPLIT_FILES.items():
        bad_df = validate_split(split_name, csv_path)

        if len(bad_df) > 0:
            all_bad_dfs.append(bad_df)

    if all_bad_dfs:
        all_bad_files = pd.concat(all_bad_dfs, ignore_index=True)
    else:
        all_bad_files = pd.DataFrame(
            columns=[
                "patientId",
                "class",
                "label",
                "triage_priority",
                "image_relative_path",
                "split",
                "error_message",
            ]
        )

    bad_report_path = PROCESSED_DATA_DIR / "bad_dicom_files.csv"
    all_bad_files.to_csv(bad_report_path, index=False)

    print("\nDICOM validation complete.")
    print(f"Total bad files: {len(all_bad_files)}")
    print(f"Bad file report saved to: {bad_report_path}")

    if len(all_bad_files) > 0:
        print("\nBad file error summary:")
        print(all_bad_files["error_message"].value_counts())


if __name__ == "__main__":
    main()