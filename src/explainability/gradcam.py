from pathlib import Path
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    CLASS_NAMES,
)
from src.data.dataset import ChestXrayDataset
from src.models.model import build_model


class GradCAM:
    """
    Grad-CAM implementation for CNN-based image classifiers.

    It works by:
    1. Saving feature maps from a target convolutional layer.
    2. Saving gradients of the target class score with respect to those feature maps.
    3. Weighting feature maps by average gradients.
    4. Producing a class-specific heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_index: int,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """
        Generate Grad-CAM heatmap for one image.

        Args:
            input_tensor:
                Shape [1, 3, 224, 224]

            target_class_index:
                Class index for which Grad-CAM should be generated.

        Returns:
            cam:
                Normalized heatmap as numpy array of shape [224, 224]

            logits:
                Raw model output.
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        target_score = logits[:, target_class_index].sum()
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].cpu().numpy()

        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, logits

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


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


def get_target_layer(model: torch.nn.Module, model_name: str):
    """
    Select the final spatial feature layer for Grad-CAM.

    For ConvNeXt-Tiny:
        model.features[-1] is the final ConvNeXt stage.

    For ResNet50:
        model.layer4[-1] is the final residual block.
    """
    if model_name == "convnext_tiny":
        return model.features[-1]

    if model_name == "resnet50":
        return model.layer4[-1]

    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def preprocess_image():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def pil_to_numpy_rgb(image):
    image = image.resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0

    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)

    return image_np


def overlay_heatmap_on_image(
    image_np: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    heatmap = plt.cm.jet(cam)[..., :3]
    overlay = (1 - alpha) * image_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def choose_target_class(
    mode: str,
    true_label: int,
    predicted_label: int,
) -> int:
    if mode == "predicted":
        return predicted_label

    if mode == "true":
        return true_label

    if mode == "lung_opacity":
        return 2

    raise ValueError(f"Unsupported target mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanation for a chest X-ray model."
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
        "--index",
        type=int,
        default=0,
        help="Row index from the CSV file.",
    )

    parser.add_argument(
        "--target-mode",
        type=str,
        default="predicted",
        choices=["predicted", "true", "lung_opacity"],
        help=(
            "predicted: explain predicted class, "
            "true: explain ground-truth class, "
            "lung_opacity: always explain high-priority class"
        ),
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="gradcam_sample",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    csv_path = PROJECT_ROOT / args.csv

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ChestXrayDataset(
        csv_path=csv_path,
        project_root=PROJECT_ROOT,
        transform=None,
    )

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(
            f"Index {args.index} is out of range for dataset length {len(dataset)}"
        )

    row = dataset.data.iloc[args.index]

    image_path = PROJECT_ROOT / row["image_relative_path"]
    true_label = int(row["label"])

    pil_image = dataset._load_dicom_image(image_path)

    transform = preprocess_image()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

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

    target_layer = get_target_layer(model, args.model_name)

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    # First forward pass to get prediction.
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = int(torch.argmax(probabilities, dim=1).item())

    target_class = choose_target_class(
        mode=args.target_mode,
        true_label=true_label,
        predicted_label=predicted_label,
    )

    cam, logits = gradcam.generate(
        input_tensor=input_tensor,
        target_class_index=target_class,
    )

    gradcam.remove_hooks()

    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    image_np = pil_to_numpy_rgb(pil_image)
    overlay = overlay_heatmap_on_image(image_np=image_np, cam=cam)

    true_class = CLASS_NAMES[true_label]
    predicted_class = CLASS_NAMES[predicted_label]
    target_class_name = CLASS_NAMES[target_class]

    print("Image path:", image_path)
    print("True class:", true_class)
    print("Predicted class:", predicted_class)
    print("Grad-CAM target class:", target_class_name)

    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{class_name}: {probability:.4f}")

    save_path = FIGURES_DIR / f"{args.output_name}_index_{args.index}_{args.target_mode}.png"

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np, cmap="gray")
    plt.title(f"Original\nTrue: {true_class}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap="jet")
    plt.title(f"Grad-CAM\nTarget: {target_class_name}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(
        f"Overlay\nPred: {predicted_class}\n"
        f"P(Lung Opacity): {probabilities[2]:.3f}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved Grad-CAM figure to:", save_path)


if __name__ == "__main__":
    main()