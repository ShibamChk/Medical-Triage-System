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
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, CLASS_NAMES
from src.data.dataset import ChestXrayDataset
from src.models.model import build_model


TARGET_CLASS_INDEX = 2
TARGET_CLASS_NAME = "Lung Opacity"


def safe_torch_load(checkpoint_path: Path, device: torch.device):
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


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    labels = np.array(all_labels)
    probabilities = np.array(all_probabilities)

    return labels, probabilities


def predict_with_lung_opacity_threshold(
    probabilities: np.ndarray,
    threshold: float,
    target_class_index: int = TARGET_CLASS_INDEX,
) -> np.ndarray:
    """
    Threshold rule:

    If P(Lung Opacity) >= threshold:
        predict Lung Opacity
    else:
        predict argmax among all classes

    This increases sensitivity for the high-priority class when threshold is lowered.
    """
    argmax_predictions = np.argmax(probabilities, axis=1)
    tuned_predictions = argmax_predictions.copy()

    target_probabilities = probabilities[:, target_class_index]
    tuned_predictions[target_probabilities >= threshold] = target_class_index

    return tuned_predictions


def get_target_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    target_class_index: int = TARGET_CLASS_INDEX,
) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[target_class_index],
        average=None,
        zero_division=0,
    )

    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "support": int(support[0]),
    }


def calculate_f_beta(
    precision: float,
    recall: float,
    beta: float = 2.0,
) -> float:
    """
    F-beta score gives more importance to recall when beta > 1.

    For triage sensitivity, F2 is useful because recall matters more than precision.
    """
    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta ** 2

    return (1 + beta_squared) * precision * recall / (
        beta_squared * precision + recall
    )


def search_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    min_precision: float,
    beta: float,
) -> pd.DataFrame:
    rows = []

    thresholds = np.arange(
        threshold_min,
        threshold_max + threshold_step,
        threshold_step,
    )

    for threshold in thresholds:
        threshold = round(float(threshold), 4)

        predictions = predict_with_lung_opacity_threshold(
            probabilities=probabilities,
            threshold=threshold,
            target_class_index=TARGET_CLASS_INDEX,
        )

        accuracy = accuracy_score(labels, predictions)
        target_metrics = get_target_class_metrics(labels, predictions)

        f_beta = calculate_f_beta(
            precision=target_metrics["precision"],
            recall=target_metrics["recall"],
            beta=beta,
        )

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "lung_opacity_precision": target_metrics["precision"],
                "lung_opacity_recall": target_metrics["recall"],
                "lung_opacity_f1": target_metrics["f1"],
                "lung_opacity_f_beta": f_beta,
                "lung_opacity_support": target_metrics["support"],
                "meets_precision_floor": target_metrics["precision"] >= min_precision,
            }
        )

    return pd.DataFrame(rows)


def choose_best_threshold(
    threshold_results: pd.DataFrame,
    min_precision: float,
) -> dict:
    """
    Choose threshold using validation results.

    Priority:
    1. Only consider thresholds that satisfy minimum precision.
    2. Among them, maximize Lung Opacity F-beta.
    3. If none satisfy minimum precision, maximize F-beta anyway.
    """
    valid_candidates = threshold_results[
        threshold_results["meets_precision_floor"] == True
    ].copy()

    if len(valid_candidates) == 0:
        print(
            f"No threshold satisfied min_precision={min_precision}. "
            "Selecting best F-beta without precision floor."
        )
        valid_candidates = threshold_results.copy()

    valid_candidates = valid_candidates.sort_values(
        by=[
            "lung_opacity_f_beta",
            "lung_opacity_recall",
            "lung_opacity_precision",
            "accuracy",
        ],
        ascending=False,
    )

    best_row = valid_candidates.iloc[0].to_dict()

    return best_row


def plot_threshold_curve(
    threshold_results: pd.DataFrame,
    selected_threshold: float,
    save_path: Path,
):
    plt.figure(figsize=(9, 6))

    plt.plot(
        threshold_results["threshold"],
        threshold_results["lung_opacity_precision"],
        marker="o",
        label="Lung Opacity Precision",
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["lung_opacity_recall"],
        marker="o",
        label="Lung Opacity Recall",
    )

    plt.plot(
        threshold_results["threshold"],
        threshold_results["lung_opacity_f_beta"],
        marker="o",
        label="Lung Opacity F2",
    )

    plt.axvline(
        selected_threshold,
        linestyle="--",
        label=f"Selected threshold = {selected_threshold:.2f}",
    )

    plt.title("Threshold Tuning for Lung Opacity Sensitivity")
    plt.xlabel("Lung Opacity Probability Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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


def build_dataloader(csv_path: Path, batch_size: int):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ChestXrayDataset(
        csv_path=csv_path,
        project_root=PROJECT_ROOT,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune Lung Opacity threshold for triage sensitivity."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="convnext_tiny",
        choices=["resnet50", "convnext_tiny"],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_convnext_tiny.pth",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="convnext_tiny_threshold_tuned",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.55,
        help="Minimum acceptable Lung Opacity precision on validation set.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="F-beta value. beta=2 prioritizes recall.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint_path = PROJECT_ROOT / args.checkpoint

    val_csv = PROCESSED_DATA_DIR / "clean_val.csv"
    test_csv = PROCESSED_DATA_DIR / "clean_test.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            "clean_val.csv or clean_test.csv not found. Run data validation first."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
    print("Validation macro F1:", checkpoint.get("val_macro_f1"))

    print("\nCollecting validation predictions...")
    val_dataset, val_loader = build_dataloader(
        csv_path=val_csv,
        batch_size=args.batch_size,
    )

    val_labels, val_probabilities = collect_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
    )

    print("Validation samples:", len(val_dataset))

    print("\nSearching thresholds on validation set...")
    threshold_results = search_thresholds(
        labels=val_labels,
        probabilities=val_probabilities,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        min_precision=args.min_precision,
        beta=args.beta,
    )

    best_threshold_info = choose_best_threshold(
        threshold_results=threshold_results,
        min_precision=args.min_precision,
    )

    selected_threshold = float(best_threshold_info["threshold"])

    print("\nSelected threshold using validation set:")
    print(best_threshold_info)

    threshold_results_path = REPORTS_DIR / f"threshold_search_{args.output_name}.csv"
    threshold_results.to_csv(threshold_results_path, index=False)

    threshold_curve_path = FIGURES_DIR / f"threshold_curve_{args.output_name}.png"

    plot_threshold_curve(
        threshold_results=threshold_results,
        selected_threshold=selected_threshold,
        save_path=threshold_curve_path,
    )

    print("\nCollecting test predictions...")
    test_dataset, test_loader = build_dataloader(
        csv_path=test_csv,
        batch_size=args.batch_size,
    )

    test_labels, test_probabilities = collect_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print("Test samples:", len(test_dataset))

    # Standard argmax predictions
    standard_predictions = np.argmax(test_probabilities, axis=1)

    # Threshold-tuned predictions
    tuned_predictions = predict_with_lung_opacity_threshold(
        probabilities=test_probabilities,
        threshold=selected_threshold,
        target_class_index=TARGET_CLASS_INDEX,
    )

    standard_report = classification_report(
        test_labels,
        standard_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    tuned_report = classification_report(
        test_labels,
        tuned_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    readable_standard_report = classification_report(
        test_labels,
        standard_predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    readable_tuned_report = classification_report(
        test_labels,
        tuned_predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    standard_cm = confusion_matrix(test_labels, standard_predictions)
    tuned_cm = confusion_matrix(test_labels, tuned_predictions)

    print("\nStandard Argmax Test Report:")
    print(readable_standard_report)

    print("\nThreshold-Tuned Test Report:")
    print(readable_tuned_report)

    print("\nStandard Confusion Matrix:")
    print(standard_cm)

    print("\nThreshold-Tuned Confusion Matrix:")
    print(tuned_cm)

    standard_accuracy = accuracy_score(test_labels, standard_predictions)
    tuned_accuracy = accuracy_score(test_labels, tuned_predictions)

    print("\nStandard Accuracy:", round(standard_accuracy, 4))
    print("Threshold-Tuned Accuracy:", round(tuned_accuracy, 4))

    results_path = REPORTS_DIR / f"threshold_tuning_results_{args.output_name}.json"

    with open(results_path, "w") as file:
        json.dump(
            {
                "model_name": args.model_name,
                "checkpoint": args.checkpoint,
                "selected_threshold": selected_threshold,
                "selection_metric": f"Validation Lung Opacity F{args.beta}",
                "min_precision": args.min_precision,
                "best_threshold_info": best_threshold_info,
                "standard_test_accuracy": standard_accuracy,
                "threshold_tuned_test_accuracy": tuned_accuracy,
                "standard_test_report": standard_report,
                "threshold_tuned_test_report": tuned_report,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_val_macro_f1": checkpoint.get("val_macro_f1"),
            },
            file,
            indent=4,
        )

    standard_cm_path = FIGURES_DIR / f"confusion_matrix_standard_{args.output_name}.png"
    tuned_cm_path = FIGURES_DIR / f"confusion_matrix_tuned_{args.output_name}.png"
    tuned_cm_norm_path = FIGURES_DIR / f"confusion_matrix_tuned_normalized_{args.output_name}.png"

    plot_confusion_matrix(
        cm=standard_cm,
        class_names=CLASS_NAMES,
        save_path=standard_cm_path,
        normalize=False,
    )

    plot_confusion_matrix(
        cm=tuned_cm,
        class_names=CLASS_NAMES,
        save_path=tuned_cm_path,
        normalize=False,
    )

    plot_confusion_matrix(
        cm=tuned_cm,
        class_names=CLASS_NAMES,
        save_path=tuned_cm_norm_path,
        normalize=True,
    )

    probabilities_df = pd.DataFrame(
        test_probabilities,
        columns=[
            "prob_Normal",
            "prob_No_Lung_Opacity_Not_Normal",
            "prob_Lung_Opacity",
        ],
    )

    predictions_df = pd.DataFrame({
        "true_label": test_labels,
        "standard_prediction": standard_predictions,
        "threshold_tuned_prediction": tuned_predictions,
        "true_class": [CLASS_NAMES[label] for label in test_labels],
        "standard_predicted_class": [
            CLASS_NAMES[prediction] for prediction in standard_predictions
        ],
        "threshold_tuned_predicted_class": [
            CLASS_NAMES[prediction] for prediction in tuned_predictions
        ],
    })

    predictions_df = pd.concat([predictions_df, probabilities_df], axis=1)

    predictions_path = REPORTS_DIR / f"threshold_tuned_predictions_{args.output_name}.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print("\nSaved threshold search to:", threshold_results_path)
    print("Saved threshold curve to:", threshold_curve_path)
    print("Saved threshold tuning results to:", results_path)
    print("Saved tuned predictions to:", predictions_path)
    print("Saved tuned confusion matrix to:", tuned_cm_path)
    print("Saved normalized tuned confusion matrix to:", tuned_cm_norm_path)


if __name__ == "__main__":
    main()