from pathlib import Path
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, FIGURES_DIR


def plot_curve(
    history: pd.DataFrame,
    x_column: str,
    y_columns: list[str],
    title: str,
    ylabel: str,
    save_path: Path,
):
    plt.figure(figsize=(8, 5))

    for column in y_columns:
        plt.plot(history[x_column], history[column], marker="o", label=column)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training curves from a training history CSV."
    )

    parser.add_argument(
        "--history",
        type=str,
        default="reports/training_history_resnet50.csv",
        help="Path to training history CSV.",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="resnet50",
        help="Name used for saving output figures.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    history_path = PROJECT_ROOT / args.history

    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(history_path)

    print("Training history:")
    print(history)

    plot_curve(
        history=history,
        x_column="epoch",
        y_columns=["train_loss", "val_loss"],
        title="Training vs Validation Loss",
        ylabel="Loss",
        save_path=FIGURES_DIR / f"loss_curve_{args.output_name}.png",
    )

    plot_curve(
        history=history,
        x_column="epoch",
        y_columns=["train_accuracy", "val_accuracy"],
        title="Training vs Validation Accuracy",
        ylabel="Accuracy",
        save_path=FIGURES_DIR / f"accuracy_curve_{args.output_name}.png",
    )

    plot_curve(
        history=history,
        x_column="epoch",
        y_columns=["val_macro_precision", "val_macro_recall", "val_macro_f1"],
        title="Validation Macro Metrics",
        ylabel="Score",
        save_path=FIGURES_DIR / f"macro_metrics_curve_{args.output_name}.png",
    )

    print("\nSaved figures:")
    print(FIGURES_DIR / f"loss_curve_{args.output_name}.png")
    print(FIGURES_DIR / f"accuracy_curve_{args.output_name}.png")
    print(FIGURES_DIR / f"macro_metrics_curve_{args.output_name}.png")


if __name__ == "__main__":
    main()