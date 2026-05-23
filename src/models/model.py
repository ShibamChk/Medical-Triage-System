import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


def build_resnet50_model(
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a ResNet50 model for chest X-ray triage classification.

    Classes:
        0 -> Normal
        1 -> No Lung Opacity / Not Normal
        2 -> Lung Opacity
    """

    weights = ResNet50_Weights.DEFAULT if pretrained else None

    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Model factory.

    Supported models:
        - resnet50
    """

    model_name = model_name.lower()

    if model_name == "resnet50":
        return build_resnet50_model(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(
        f"Unsupported model_name: {model_name}. "
        "Currently supported: 'resnet50'."
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )


def count_total_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())