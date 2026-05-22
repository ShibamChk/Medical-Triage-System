from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torch.utils.data import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, ExplicitVRLittleEndian


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for RSNA chest X-ray triage classification.

    This Dataset expects a CSV containing:
        - image_relative_path
        - label

    Each item returns:
        image_tensor: torch.Tensor
        label: torch.Tensor

    Classes:
        0 -> Normal
        1 -> No Lung Opacity / Not Normal
        2 -> Lung Opacity
    """

    def __init__(
        self,
        csv_path: str | Path,
        project_root: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.data = pd.read_csv(self.csv_path)

        required_columns = ["image_relative_path", "label"]
        for column in required_columns:
            if column not in self.data.columns:
                raise ValueError(
                    f"Missing required column '{column}' in {self.csv_path}"
                )

        self.data["image_relative_path"] = self.data["image_relative_path"].astype(str)
        self.data["label"] = self.data["label"].astype(int)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[index]

        image_path = self.project_root / row["image_relative_path"]
        label = int(row["label"])

        image = self._load_dicom_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def _load_dicom_image(self, image_path: Path) -> Image.Image:
        """
        Load a valid DICOM image and convert it into a PIL RGB image.

        Important:
        Bad DICOM files should be removed before training using:
            python src/data/validate_dicom_files.py

        This method still handles common metadata problems:
        - missing DICOM preamble/header
        - missing TransferSyntaxUID
        - MONOCHROME1 inverted grayscale
        """
        if not image_path.exists():
            raise FileNotFoundError(f"DICOM image not found: {image_path}")

        dicom = pydicom.dcmread(str(image_path), force=True)

        image = self._extract_pixel_array(dicom=dicom, image_path=image_path)

        slope = float(getattr(dicom, "RescaleSlope", 1.0))
        intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        photometric_interpretation = getattr(dicom, "PhotometricInterpretation", "")

        if photometric_interpretation == "MONOCHROME1":
            image = image.max() - image

        image = image - image.min()

        if image.max() > 0:
            image = image / image.max()

        image = (image * 255).astype(np.uint8)

        image = Image.fromarray(image)
        image = image.convert("RGB")

        return image

    def _extract_pixel_array(
        self,
        dicom: pydicom.dataset.FileDataset,
        image_path: Path,
    ) -> np.ndarray:
        """
        Extract pixel data from DICOM.

        Metadata issues can sometimes be fixed.
        Missing pixel data cannot be fixed and must be removed by validation.
        """
        if not hasattr(dicom, "file_meta") or dicom.file_meta is None:
            dicom.file_meta = FileMetaDataset()

        if not self._has_pixel_data(dicom):
            raise RuntimeError(
                f"DICOM file has no pixel data and must be removed: {image_path}"
            )

        transfer_syntaxes_to_try = [
            None,
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
        ]

        last_error = None

        for transfer_syntax in transfer_syntaxes_to_try:
            try:
                if transfer_syntax is not None:
                    dicom.file_meta.TransferSyntaxUID = transfer_syntax

                if hasattr(dicom, "_pixel_array"):
                    delattr(dicom, "_pixel_array")

                return dicom.pixel_array.astype(np.float32)

            except Exception as error:
                last_error = error

        raise RuntimeError(
            f"Failed to extract pixel array from DICOM file: {image_path}"
        ) from last_error

    @staticmethod
    def _has_pixel_data(dicom: pydicom.dataset.FileDataset) -> bool:
        return (
            "PixelData" in dicom
            or "FloatPixelData" in dicom
            or "DoubleFloatPixelData" in dicom
        )