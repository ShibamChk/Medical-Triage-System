from pathlib import Path
import sys
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.data.dataset import ChestXrayDataset
from src.models.model import build_model
from src.evaluation.metrics import calculate_accuracy


def set_seed(seed: int = 42):
    """
    Make experiments more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_csv: Path, num_classes: int = 3) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.

    We calculate this even if we do not use it immediately,
    because it helps us inspect class imbalance.
    """
    df = pd.read_csv(train_csv)
    class_counts = df["label"].value_counts().sort_index()
    total_samples = len(df)

    weights = []

    for class_idx in range(num_classes):
        class_count = class_counts.get(class_idx, 0)

        if class_count == 0:
            raise ValueError(f"Class {class_idx} has zero samples.")

        weight = total_samples / (num_classes * class_count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_accuracy = 0.0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        running_accuracy += calculate_accuracy(outputs.detach(), labels) * batch_size

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)

    return epoch_loss, epoch_accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float, float, float]:
    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size
            running_accuracy += calculate_accuracy(outputs, labels) * batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return epoch_loss, epoch_accuracy, macro_precision, macro_recall, macro_f1


def main():
    set_seed(42)

    # -----------------------------
    # Training configuration
    # -----------------------------
    model_name = "resnet50"
    num_classes = 3
    pretrained = True
    freeze_backbone = False

    batch_size = 16
    num_epochs = 20
    early_stopping_patience = 5
    min_delta = 1e-4

    learning_rate = 1e-4
    weight_decay = 1e-4

    use_class_weights = False

    # On Windows, keep num_workers=0 first for stability.
    num_workers = 0

    # -----------------------------
    # Device setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"

    print("Using device:", device)
    print("Mixed precision enabled:", use_amp)
    print("Training model:", model_name)

    # -----------------------------
    # Paths
    # -----------------------------
    train_csv = PROCESSED_DATA_DIR / "clean_train.csv"
    val_csv = PROCESSED_DATA_DIR / "clean_val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            "Clean split files not found.\n"
            "Run these first:\n"
            "python src/data/create_metadata.py\n"
            "python src/data/create_splits.py\n"
            "python src/data/validate_dicom_files.py"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best_model_path = MODELS_DIR / f"best_{model_name}.pth"
    history_path = REPORTS_DIR / f"training_history_{model_name}.csv"

    # -----------------------------
    # Transforms
    # -----------------------------
    # Medical-image-safe augmentations:
    # - Small rotation only
    # - No aggressive crop
    # - No horizontal flip initially because laterality can matter in medical imaging
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=7),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # -----------------------------
    # Dataset and DataLoader
    # -----------------------------
    train_dataset = ChestXrayDataset(
        csv_path=train_csv,
        project_root=PROJECT_ROOT,
        transform=train_transform,
    )

    val_dataset = ChestXrayDataset(
        csv_path=val_csv,
        project_root=PROJECT_ROOT,
        transform=val_transform,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # -----------------------------
    # Model
    # -----------------------------
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    model = model.to(device)

    # -----------------------------
    # Loss, optimizer, scheduler
    # -----------------------------
    class_weights = compute_class_weights(
        train_csv=train_csv,
        num_classes=num_classes,
    ).to(device)

    print("Class weights reference:", class_weights)
    print("Using class weights in loss:", use_class_weights)

    if use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    scaler = GradScaler(enabled=use_amp)

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    training_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        (
            val_loss,
            val_accuracy,
            val_macro_precision,
            val_macro_recall,
            val_macro_f1,
        ) = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_accuracy:.4f}")
        print(
            f"Val Precision: {val_macro_precision:.4f} | "
            f"Val Recall: {val_macro_recall:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )
        print(f"Learning Rate: {current_lr:.8f}")

        training_history.append(
            {
                "epoch": epoch + 1,
                "model_name": model_name,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_precision": val_macro_precision,
                "val_macro_recall": val_macro_recall,
                "val_macro_f1": val_macro_f1,
                "learning_rate": current_lr,
                "use_class_weights": use_class_weights,
            }
        )

        if val_macro_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_macro_f1
            epochs_without_improvement = 0

            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_name": model_name,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_macro_precision": val_macro_precision,
                    "val_macro_recall": val_macro_recall,
                    "val_macro_f1": val_macro_f1,
                    "class_weights": class_weights.detach().cpu(),
                    "use_class_weights": use_class_weights,
                },
                best_model_path,
            )

            print(f"Best model saved to: {best_model_path}")

        else:
            epochs_without_improvement += 1
            print(
                f"No macro F1 improvement for "
                f"{epochs_without_improvement}/{early_stopping_patience} epochs."
            )

        if epochs_without_improvement >= early_stopping_patience:
            print("\nEarly stopping triggered.")
            break

    history_df = pd.DataFrame(training_history)
    history_df.to_csv(history_path, index=False)

    print("\nTraining complete.")
    print(f"Best validation macro F1: {best_val_f1:.4f}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()