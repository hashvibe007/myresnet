import torch
import yaml
from src.models.resnet import ResNet50
from src.data.dataset import ImageNetDataModule
from torch.cuda.amp import GradScaler


def test_model_accuracy():
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = ResNet50(num_classes=config["model"]["num_classes"])

    # Load trained weights
    checkpoint = torch.load("best_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create scaler for mixed precision (if needed for inference)
    scaler = GradScaler()
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    model.eval()

    # Initialize data
    data_module = ImageNetDataModule(config)
    data_module.setup()
    val_loader = data_module.val_dataloader()

    # Test accuracy
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Assert accuracy is above 70%
    assert accuracy >= 70.0, f"Model accuracy ({accuracy:.2f}%) is below 70%"
