from pathlib import Path
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, CLASS_NAMES
from src.data.dataset import ChestXrayDataset
from src.models.model import build_model


def safe_torch_load(checkpoint_path: Path, device: torch.device):
    """
    Load checkpoint safely across PyTorch versions.
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


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    average_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    normalize: bool = False,
):
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        title = "Normalized Confusion Matrix"
    else:
        cm_display = cm
        title = "Confusion Matrix"

    plt.figure(figsize=(9, 7))
    plt.imshow(cm_display)
    plt.title(title)
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30, ha="right")
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_display[i, j]:.2f}" if normalize else str(cm[i, j])
            plt.text(j, i, text, ha="center", va="center")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained chest X-ray triage model."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        choices=["resnet50", "convnext_tiny"],
        help="Model architecture to evaluate.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_resnet50.pth",
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="resnet50",
        help="Name used for report files.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_csv = PROCESSED_DATA_DIR / "clean_test.csv"
    checkpoint_path = PROJECT_ROOT / args.checkpoint

    if not test_csv.exists():
        raise FileNotFoundError(
            "clean_test.csv not found. Run DICOM validation first."
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = ChestXrayDataset(
        csv_path=test_csv,
        project_root=PROJECT_ROOT,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print("Test samples:", len(test_dataset))

    model = build_model(
        model_name=args.model_name,
        num_classes=3,
        pretrained=False,
        freeze_backbone=False,
    )

    checkpoint = safe_torch_load(checkpoint_path, device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print("\nLoaded checkpoint:")
    print("Epoch:", checkpoint.get("epoch"))
    print("Validation loss:", checkpoint.get("val_loss"))
    print("Validation accuracy:", checkpoint.get("val_accuracy"))
    print("Validation macro precision:", checkpoint.get("val_macro_precision"))
    print("Validation macro recall:", checkpoint.get("val_macro_recall"))
    print("Validation macro F1:", checkpoint.get("val_macro_f1"))

    criterion = nn.CrossEntropyLoss()

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    labels = results["labels"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]

    print("\nTest Loss:", round(results["loss"], 4))
    print("Test Accuracy:", round(results["accuracy"], 4))

    readable_report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(readable_report)

    cm = confusion_matrix(labels, predictions)

    print("\nConfusion Matrix:")
    print(cm)

    metrics_path = REPORTS_DIR / f"test_metrics_{args.output_name}.json"

    with open(metrics_path, "w") as file:
        json.dump(
            {
                "model_name": args.model_name,
                "output_name": args.output_name,
                "test_loss": results["loss"],
                "test_accuracy": results["accuracy"],
                "classification_report": report,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_val_loss": checkpoint.get("val_loss"),
                "checkpoint_val_accuracy": checkpoint.get("val_accuracy"),
                "checkpoint_val_macro_precision": checkpoint.get("val_macro_precision"),
                "checkpoint_val_macro_recall": checkpoint.get("val_macro_recall"),
                "checkpoint_val_macro_f1": checkpoint.get("val_macro_f1"),
            },
            file,
            indent=4,
        )

    raw_cm_path = FIGURES_DIR / f"confusion_matrix_{args.output_name}.png"
    normalized_cm_path = FIGURES_DIR / f"confusion_matrix_normalized_{args.output_name}.png"

    plot_confusion_matrix(cm, CLASS_NAMES, raw_cm_path, normalize=False)
    plot_confusion_matrix(cm, CLASS_NAMES, normalized_cm_path, normalize=True)

    probabilities_df = pd.DataFrame(
        probabilities,
        columns=[
            "prob_Normal",
            "prob_No_Lung_Opacity_Not_Normal",
            "prob_Lung_Opacity",
        ],
    )

    predictions_df = pd.DataFrame({
        "true_label": labels,
        "predicted_label": predictions,
        "true_class": [CLASS_NAMES[label] for label in labels],
        "predicted_class": [CLASS_NAMES[pred] for pred in predictions],
    })

    predictions_df = pd.concat([predictions_df, probabilities_df], axis=1)

    predictions_path = REPORTS_DIR / f"test_predictions_{args.output_name}.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print("\nSaved metrics to:", metrics_path)
    print("Saved raw confusion matrix to:", raw_cm_path)
    print("Saved normalized confusion matrix to:", normalized_cm_path)
    print("Saved predictions to:", predictions_path)


if __name__ == "__main__":
    main()