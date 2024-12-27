import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from src.models.resnet import ResNet50
from src.data.dataset import ImageNetDataModule
from torch.cuda.amp import autocast, GradScaler


def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Initialize progress bar
    pbar = tqdm(train_loader, desc="Training", leave=True)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{running_loss/total:.3f}", "acc": f"{acc:.2f}%"})

    return 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    # Initialize progress bar for validation
    pbar = tqdm(val_loader, desc="Validation", leave=True)

    with torch.no_grad(), autocast():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            acc = 100.0 * correct / total
            pbar.set_postfix({"loss": f"{val_loss/total:.3f}", "acc": f"{acc:.2f}%"})

    accuracy = 100.0 * correct / total
    avg_loss = val_loss / len(val_loader)
    print(f"\nValidation - Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%")
    return accuracy


def main(args):
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Update config with command line arguments
    if args.data_path:
        config["data"]["root"] = args.data_path
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()

    # Initialize data module
    data_module = ImageNetDataModule(config)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Initialize model
    model = ResNet50(num_classes=config["model"]["num_classes"])
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["training"]["lr_schedule"]["milestones"],
        gamma=config["training"]["lr_schedule"]["gamma"],
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {"train_acc": [], "val_acc": [], "best_acc": 0}

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 50)

        train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "accuracy": val_acc,
            "history": history,
        }

        # Save latest checkpoint
        torch.save(checkpoint, output_dir / "latest_model.pth")

        # Save best model
        if val_acc > history["best_acc"]:
            history["best_acc"] = val_acc
            torch.save(checkpoint, output_dir / "best_model.pth")
            print(f"Saved new best model with accuracy: {val_acc:.2f}%")

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {history['best_acc']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 on ImageNet")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--data_path", type=str, help="Path to ImageNet dataset (overrides config)"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save model checkpoints",
    )

    args = parser.parse_args()
    main(args)
