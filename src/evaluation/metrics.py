import torch


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate batch-level classification accuracy.

    Args:
        outputs:
            Raw logits of shape [batch_size, num_classes]

        labels:
            Ground-truth class labels of shape [batch_size]
    """
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)

    return correct / total