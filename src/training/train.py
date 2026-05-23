from pathlib import Path
import sys
import copy
import random
import argparse

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


def safe_torch_load(checkpoint_path: Path, device: torch.device):
    """
    Load checkpoints safely across different PyTorch versions.
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


def compute_class_weights(train_csv: Path, num_classes: int = 3) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.

    This is useful for class imbalance analysis and optional weighted loss.
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


def save_training_history(
    training_history: list[dict],
    history_path: Path,
):
    """
    Save training history after every epoch.

    This protects progress if the PC shuts down suddenly.
    """
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(history_path, index=False)


def save_latest_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    best_val_f1: float,
    epochs_without_improvement: int,
    val_loss: float,
    val_accuracy: float,
    val_macro_precision: float,
    val_macro_recall: float,
    val_macro_f1: float,
    class_weights: torch.Tensor,
    use_class_weights: bool,
):
    """
    Save latest checkpoint after every epoch.

    This checkpoint is used for resuming interrupted training.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_f1": best_val_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_precision": val_macro_precision,
            "val_macro_recall": val_macro_recall,
            "val_macro_f1": val_macro_f1,
            "class_weights": class_weights.detach().cpu(),
            "use_class_weights": use_class_weights,
        },
        checkpoint_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a chest X-ray triage model."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="convnext_tiny",
        choices=["resnet50", "convnext_tiny"],
        help="Model architecture to train.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class-weighted CrossEntropyLoss.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(42)

    # -----------------------------
    # Training configuration
    # -----------------------------
    model_name = args.model_name
    num_classes = 3
    pretrained = True
    freeze_backbone = False

    batch_size = args.batch_size
    num_epochs = args.epochs
    early_stopping_patience = args.patience
    min_delta = 1e-4

    learning_rate = args.lr
    weight_decay = args.weight_decay

    use_class_weights = args.use_class_weights

    # On Windows, num_workers=0 is the most stable.
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
    print("Resume training:", args.resume)

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
    latest_model_path = MODELS_DIR / f"latest_{model_name}.pth"
    history_path = REPORTS_DIR / f"training_history_{model_name}.csv"

    # -----------------------------
    # Transforms
    # -----------------------------
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
    # Loss, optimizer, scheduler, scaler
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

    scaler = GradScaler(device=device.type, enabled=use_amp)

    # -----------------------------
    # Resume state
    # -----------------------------
    start_epoch = 0
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    training_history = []

    if args.resume:
        if not latest_model_path.exists():
            raise FileNotFoundError(
                f"Cannot resume because latest checkpoint was not found: {latest_model_path}"
            )

        print(f"Loading latest checkpoint from: {latest_model_path}")
        checkpoint = safe_torch_load(latest_model_path, device)

        checkpoint_model_name = checkpoint.get("model_name")

        if checkpoint_model_name != model_name:
            raise ValueError(
                f"Checkpoint model_name is {checkpoint_model_name}, "
                f"but requested model_name is {model_name}."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint["epoch"])
        best_val_f1 = float(checkpoint.get("best_val_f1", 0.0))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )

        if history_path.exists():
            history_df = pd.read_csv(history_path)
            training_history = history_df.to_dict("records")

        print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Best validation macro F1 so far: {best_val_f1:.4f}")

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch_index in range(start_epoch, num_epochs):
        epoch_number = epoch_index + 1

        print(f"\nEpoch {epoch_number}/{num_epochs}")

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

        epoch_record = {
            "epoch": epoch_number,
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

        # Remove duplicate epoch row if resuming and overwriting same epoch.
        training_history = [
            row for row in training_history if int(row["epoch"]) != epoch_number
        ]
        training_history.append(epoch_record)

        # Save history after every epoch.
        save_training_history(training_history, history_path)

        # Best checkpoint logic.
        if val_macro_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_macro_f1
            epochs_without_improvement = 0

            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": epoch_number,
                    "model_name": model_name,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_f1": best_val_f1,
                    "epochs_without_improvement": epochs_without_improvement,
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

        # Always save latest checkpoint after every epoch.
        save_latest_checkpoint(
            checkpoint_path=latest_model_path,
            epoch=epoch_number,
            model_name=model_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_f1=best_val_f1,
            epochs_without_improvement=epochs_without_improvement,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_macro_precision=val_macro_precision,
            val_macro_recall=val_macro_recall,
            val_macro_f1=val_macro_f1,
            class_weights=class_weights,
            use_class_weights=use_class_weights,
        )

        print(f"Latest checkpoint saved to: {latest_model_path}")
        print(f"Training history saved to: {history_path}")

        if epochs_without_improvement >= early_stopping_patience:
            print("\nEarly stopping triggered.")
            break

    print("\nTraining complete.")
    print(f"Best validation macro F1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Latest checkpoint: {latest_model_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()