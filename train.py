import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from data_loader import get_dataloaders
import config


def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    train_loader, val_loader = get_dataloaders()

    model = SimpleCNN(num_classes=config.NUM_CLASSES)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{config.EPOCHS}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'simple_cnn.pth'))
    print('Model saved to saved_models/simple_cnn.pth')


if __name__ == '__main__':
    train()