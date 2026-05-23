from pathlib import Path
import sys
import json
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, ExplicitVRLittleEndian


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    PROJECT_ROOT,
    CLASS_NAMES,
    LABEL_TO_CLASS,
    TRIAGE_MAPPING,
)
from src.models.model import build_model


def safe_torch_load(checkpoint_path: Path, device: torch.device):
    """
    Safely load a PyTorch checkpoint across different PyTorch versions.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def has_pixel_data(dicom: pydicom.dataset.FileDataset) -> bool:
    """
    Check whether a DICOM object contains pixel data.
    """
    return (
        "PixelData" in dicom
        or "FloatPixelData" in dicom
        or "DoubleFloatPixelData" in dicom
    )


def extract_dicom_pixel_array(
    dicom: pydicom.dataset.FileDataset,
    image_path: Path,
) -> np.ndarray:
    """
    Extract pixel array from a DICOM file.

    This handles common DICOM metadata issues:
    - missing file_meta
    - missing TransferSyntaxUID
    - incomplete preamble/header

    It cannot fix files that truly have no pixel data.
    """
    if not hasattr(dicom, "file_meta") or dicom.file_meta is None:
        dicom.file_meta = FileMetaDataset()

    if not has_pixel_data(dicom):
        raise RuntimeError(
            f"DICOM file has no pixel data and cannot be used: {image_path}"
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


def load_dicom_as_pil(image_path: Path) -> Image.Image:
    """
    Load a DICOM image and convert it into a PIL RGB image.
    """
    dicom = pydicom.dcmread(str(image_path), force=True)

    image = extract_dicom_pixel_array(
        dicom=dicom,
        image_path=image_path,
    )

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

    pil_image = Image.fromarray(image)
    pil_image = pil_image.convert("RGB")

    return pil_image


def load_image_as_pil(image_path: str | Path) -> Image.Image:
    """
    Load an image file as a PIL RGB image.

    Supports:
    - DICOM files: .dcm
    - Standard image files: .png, .jpg, .jpeg

    For this project, DICOM is the primary expected input.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_path.suffix.lower() == ".dcm":
        return load_dicom_as_pil(image_path)

    pil_image = Image.open(image_path)
    pil_image = pil_image.convert("RGB")

    return pil_image


def get_inference_transform():
    """
    Inference preprocessing.

    This must match validation/test preprocessing used during training.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_trained_model(
    model_name: str,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a trained model checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(
        model_name=model_name,
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_backbone=False,
    )

    checkpoint = safe_torch_load(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def apply_lung_opacity_threshold(
    probabilities: np.ndarray,
    threshold: Optional[float],
) -> tuple[int, str]:
    """
    Apply optional Lung Opacity sensitivity threshold.

    If threshold is None:
        standard argmax prediction is used.

    If threshold is provided:
        if P(Lung Opacity) >= threshold:
            predict Lung Opacity
        else:
            use standard argmax prediction
    """
    argmax_label = int(np.argmax(probabilities))

    if threshold is None:
        return argmax_label, "argmax"

    lung_opacity_probability = float(probabilities[2])

    if lung_opacity_probability >= threshold:
        return 2, f"lung_opacity_threshold_{threshold}"

    return argmax_label, f"argmax_with_lung_opacity_threshold_{threshold}"


def predict_image(
    image_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    lung_opacity_threshold: Optional[float] = None,
) -> dict:
    """
    Run prediction on one image.

    Returns a dictionary containing:
    - predicted class
    - triage priority
    - probability scores
    - decision mode
    """
    image_path = Path(image_path)

    pil_image = load_image_as_pil(image_path)

    transform = get_inference_transform()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities_tensor = F.softmax(logits, dim=1)[0]

    probabilities = probabilities_tensor.detach().cpu().numpy()

    standard_argmax_label = int(np.argmax(probabilities))

    predicted_label, decision_mode = apply_lung_opacity_threshold(
        probabilities=probabilities,
        threshold=lung_opacity_threshold,
    )

    predicted_class = LABEL_TO_CLASS[predicted_label]
    standard_argmax_class = LABEL_TO_CLASS[standard_argmax_label]

    probability_scores = {
        CLASS_NAMES[index]: float(probabilities[index])
        for index in range(len(CLASS_NAMES))
    }

    result = {
        "image_path": str(image_path),
        "predicted_label": predicted_label,
        "predicted_class": predicted_class,
        "triage_priority": TRIAGE_MAPPING[predicted_class],
        "confidence": float(probabilities[predicted_label]),
        "probabilities": probability_scores,
        "standard_argmax_label": standard_argmax_label,
        "standard_argmax_class": standard_argmax_class,
        "lung_opacity_threshold": lung_opacity_threshold,
        "decision_mode": decision_mode,
    }

    return result


def predict_from_checkpoint(
    image_path: str | Path,
    model_name: str,
    checkpoint_path: str | Path,
    lung_opacity_threshold: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Convenience function:
    load model checkpoint and predict one image.

    This is useful for Streamlit/FastAPI later.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    result = predict_image(
        image_path=image_path,
        model=model,
        device=device,
        lung_opacity_threshold=lung_opacity_threshold,
    )

    result["model_name"] = model_name
    result["checkpoint_path"] = str(checkpoint_path)
    result["device"] = str(device)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a chest X-ray image."
    )

    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to DICOM or image file.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="convnext_tiny",
        choices=["convnext_tiny", "resnet50"],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_convnext_tiny.pth",
    )

    parser.add_argument(
        "--lung-opacity-threshold",
        type=float,
        default=None,
        help=(
            "Optional Lung Opacity threshold. "
            "Example: 0.29 for high-sensitivity triage mode."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save prediction result as JSON.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    image_path = PROJECT_ROOT / args.image_path

    result = predict_from_checkpoint(
        image_path=image_path,
        model_name=args.model_name,
        checkpoint_path=checkpoint_path,
        lung_opacity_threshold=args.lung_opacity_threshold,
    )

    print(json.dumps(result, indent=4))

    if args.output_json is not None:
        output_path = PROJECT_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as file:
            json.dump(result, file, indent=4)

        print(f"\nSaved prediction JSON to: {output_path}")


if __name__ == "__main__":
    main()