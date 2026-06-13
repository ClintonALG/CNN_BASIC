import matplotlib.pyplot as plt
import numpy as np
import torch
from data_loader import get_dataloaders
import config


def show_sample_images(num_samples=10):
    train_loader, _ = get_dataloaders()
    images, labels = next(iter(train_loader))

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f'Class: {labels[i].item()}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss over epochs')

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Accuracy over epochs')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    show_sample_images()