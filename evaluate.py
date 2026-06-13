import os
import torch
import torch.nn as nn
from model import SimpleCNN
from data_loader import get_test_loader
import config


def evaluate():
    model = SimpleCNN(num_classes=config.NUM_CLASSES)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_path = os.path.join(config.MODEL_DIR, 'simple_cnn.pth')
    if not os.path.exists(model_path):
        print('No trained model found. Train first using train.py')
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = get_test_loader()
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_loss = test_loss / len(test_loader)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    evaluate()