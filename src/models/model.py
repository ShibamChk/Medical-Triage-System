import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


def build_resnet50_model(
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a ResNet50 model for chest X-ray triage classification.
    """

    weights = ResNet50_Weights.DEFAULT if pretrained else None

    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_convnext_tiny_model(
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a ConvNeXt-Tiny model for chest X-ray triage classification.

    ConvNeXt is a modern CNN architecture inspired by transformer-era design
    choices while preserving CNN efficiency and stability.
    """

    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None

    model = convnext_tiny(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # torchvision ConvNeXt classifier:
    # classifier = [LayerNorm2d, Flatten, Linear]
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

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
        - convnext_tiny
    """

    model_name = model_name.lower()

    if model_name == "resnet50":
        return build_resnet50_model(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if model_name == "convnext_tiny":
        return build_convnext_tiny_model(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(
        f"Unsupported model_name: {model_name}. "
        "Currently supported: 'resnet50', 'convnext_tiny'."
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )


def count_total_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())