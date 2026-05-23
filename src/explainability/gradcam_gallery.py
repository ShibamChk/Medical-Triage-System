from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, FIGURES_DIR, CLASS_NAMES
from src.data.dataset import ChestXrayDataset
from src.models.model import build_model
from src.explainability.gradcam import (
    GradCAM,
    safe_torch_load,
    get_target_layer,
    preprocess_image,
    pil_to_numpy_rgb,
    overlay_heatmap_on_image,
)


PROBABILITY_COLUMNS = {
    0: "prob_Normal",
    1: "prob_No_Lung_Opacity_Not_Normal",
    2: "prob_Lung_Opacity",
}


CASE_DEFINITIONS = [
    {
        "case_name": "correct_lung_opacity",
        "display_name": "Correct Lung Opacity",
        "true_label": 2,
        "predicted_label": 2,
        "target_class": 2,
        "sort_probability_class": 2,
    },
    {
        "case_name": "missed_lung_opacity_as_medium",
        "display_name": "Missed Lung Opacity as Medium",
        "true_label": 2,
        "predicted_label": 1,
        "target_class": 2,
        "sort_probability_class": 2,
    },
    {
        "case_name": "false_positive_lung_opacity",
        "display_name": "False Positive Lung Opacity",
        "true_label": None,
        "predicted_label": 2,
        "target_class": 2,
        "sort_probability_class": 2,
    },
    {
        "case_name": "correct_normal",
        "display_name": "Correct Normal",
        "true_label": 0,
        "predicted_label": 0,
        "target_class": 0,
        "sort_probability_class": 0,
    },
    {
        "case_name": "correct_medium",
        "display_name": "Correct Medium Priority",
        "true_label": 1,
        "predicted_label": 1,
        "target_class": 1,
        "sort_probability_class": 1,
    },
]


def select_representative_samples(
    predictions_df: pd.DataFrame,
    max_per_case: int = 1,
) -> list[dict]:
    """
    Select representative examples from prediction CSV.

    The prediction CSV must align row-by-row with clean_test.csv.
    """
    required_columns = [
        "true_label",
        "predicted_label",
        "true_class",
        "predicted_class",
    ]

    for column in required_columns:
        if column not in predictions_df.columns:
            raise ValueError(f"Missing required column in predictions CSV: {column}")

    selected_samples = []

    for case in CASE_DEFINITIONS:
        case_df = predictions_df.copy()

        if case["true_label"] is not None:
            case_df = case_df[case_df["true_label"] == case["true_label"]]
        else:
            # For false positives, true class should not be Lung Opacity.
            case_df = case_df[case_df["true_label"] != 2]

        case_df = case_df[case_df["predicted_label"] == case["predicted_label"]]

        probability_column = PROBABILITY_COLUMNS.get(case["sort_probability_class"])

        if probability_column in case_df.columns:
            case_df = case_df.sort_values(
                by=probability_column,
                ascending=False,
            )

        if len(case_df) == 0:
            print(f"No sample found for case: {case['display_name']}")
            continue

        for row_index, row in case_df.head(max_per_case).iterrows():
            selected_samples.append(
                {
                    "dataset_index": int(row_index),
                    "case_name": case["case_name"],
                    "display_name": case["display_name"],
                    "target_class": case["target_class"],
                    "true_label": int(row["true_label"]),
                    "predicted_label": int(row["predicted_label"]),
                }
            )

    return selected_samples


def generate_gradcam_for_sample(
    model: torch.nn.Module,
    gradcam: GradCAM,
    dataset: ChestXrayDataset,
    dataset_index: int,
    target_class: int,
    device: torch.device,
) -> dict:
    """
    Generate Grad-CAM visualization data for one dataset sample.
    """
    row = dataset.data.iloc[dataset_index]

    image_path = PROJECT_ROOT / row["image_relative_path"]
    true_label = int(row["label"])

    pil_image = dataset._load_dicom_image(image_path)

    transform = preprocess_image()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        predicted_label = int(np.argmax(probabilities))

    cam, logits_for_cam = gradcam.generate(
        input_tensor=input_tensor,
        target_class_index=target_class,
    )

    probabilities_for_cam = (
        torch.softmax(logits_for_cam, dim=1)[0]
        .detach()
        .cpu()
        .numpy()
    )

    image_np = pil_to_numpy_rgb(pil_image)
    overlay = overlay_heatmap_on_image(
        image_np=image_np,
        cam=cam,
        alpha=0.45,
    )

    return {
        "dataset_index": dataset_index,
        "image_path": image_path,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "target_class": target_class,
        "true_class": CLASS_NAMES[true_label],
        "predicted_class": CLASS_NAMES[predicted_label],
        "target_class_name": CLASS_NAMES[target_class],
        "probabilities": probabilities_for_cam,
        "image_np": image_np,
        "cam": cam,
        "overlay": overlay,
    }


def save_individual_gradcam_figure(
    sample_result: dict,
    display_name: str,
    output_name: str,
):
    """
    Save a single 3-panel Grad-CAM figure.
    """
    dataset_index = sample_result["dataset_index"]

    save_path = (
        FIGURES_DIR
        / f"{output_name}_{display_name.replace(' ', '_').lower()}_index_{dataset_index}.png"
    )

    probabilities = sample_result["probabilities"]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(sample_result["image_np"], cmap="gray")
    plt.title(
        f"Original\n"
        f"True: {sample_result['true_class']}"
    )
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sample_result["cam"], cmap="jet")
    plt.title(
        f"Grad-CAM\n"
        f"Target: {sample_result['target_class_name']}"
    )
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(sample_result["overlay"])
    plt.title(
        f"{display_name}\n"
        f"Pred: {sample_result['predicted_class']}\n"
        f"P(Lung Opacity): {probabilities[2]:.3f}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


def save_gallery_figure(
    sample_results: list[dict],
    output_name: str,
):
    """
    Save one combined Grad-CAM gallery.

    Each row:
        Original | Grad-CAM | Overlay
    """
    if len(sample_results) == 0:
        raise ValueError("No Grad-CAM samples available for gallery.")

    rows = len(sample_results)
    cols = 3

    plt.figure(figsize=(15, 4.5 * rows))

    for row_idx, item in enumerate(sample_results):
        sample_result = item["result"]
        display_name = item["display_name"]

        probabilities = sample_result["probabilities"]

        base_position = row_idx * cols

        plt.subplot(rows, cols, base_position + 1)
        plt.imshow(sample_result["image_np"], cmap="gray")
        plt.title(
            f"{display_name}\n"
            f"Original\n"
            f"True: {sample_result['true_class']}"
        )
        plt.axis("off")

        plt.subplot(rows, cols, base_position + 2)
        plt.imshow(sample_result["cam"], cmap="jet")
        plt.title(
            f"Grad-CAM\n"
            f"Target: {sample_result['target_class_name']}"
        )
        plt.axis("off")

        plt.subplot(rows, cols, base_position + 3)
        plt.imshow(sample_result["overlay"])
        plt.title(
            f"Overlay\n"
            f"Pred: {sample_result['predicted_class']}\n"
            f"P(Lung Opacity): {probabilities[2]:.3f}"
        )
        plt.axis("off")

    plt.tight_layout()

    save_path = FIGURES_DIR / f"{output_name}_gradcam_gallery.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved gallery:", save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM explainability gallery."
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
        "--csv",
        type=str,
        default="data/processed/clean_test.csv",
    )

    parser.add_argument(
        "--predictions",
        type=str,
        default="reports/test_predictions_convnext_tiny.csv",
        help="Prediction CSV generated by evaluate.py.",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="convnext_tiny",
    )

    parser.add_argument(
        "--max-per-case",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    csv_path = PROJECT_ROOT / args.csv
    predictions_path = PROJECT_ROOT / args.predictions

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions CSV not found: {predictions_path}\n"
            "Run evaluate.py first to generate test predictions."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.read_csv(predictions_path)

    selected_samples = select_representative_samples(
        predictions_df=predictions_df,
        max_per_case=args.max_per_case,
    )

    if len(selected_samples) == 0:
        raise RuntimeError("No representative samples found.")

    print("\nSelected samples:")
    for sample in selected_samples:
        print(sample)

    dataset = ChestXrayDataset(
        csv_path=csv_path,
        project_root=PROJECT_ROOT,
        transform=None,
    )

    model = build_model(
        model_name=args.model_name,
        num_classes=3,
        pretrained=False,
        freeze_backbone=False,
    )

    checkpoint = safe_torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    target_layer = get_target_layer(
        model=model,
        model_name=args.model_name,
    )

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    gallery_items = []

    for sample in selected_samples:
        result = generate_gradcam_for_sample(
            model=model,
            gradcam=gradcam,
            dataset=dataset,
            dataset_index=sample["dataset_index"],
            target_class=sample["target_class"],
            device=device,
        )

        save_individual_gradcam_figure(
            sample_result=result,
            display_name=sample["display_name"],
            output_name=args.output_name,
        )

        gallery_items.append(
            {
                "display_name": sample["display_name"],
                "result": result,
            }
        )

    gradcam.remove_hooks()

    save_gallery_figure(
        sample_results=gallery_items,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()