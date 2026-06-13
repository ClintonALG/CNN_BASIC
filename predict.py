import os
import cv2
import numpy as np
import torch
from model import SimpleCNN
import config


def predict_image(image_path):
    model = SimpleCNN(num_classes=config.NUM_CLASSES)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_path = os.path.join(config.MODEL_DIR, 'simple_cnn.pth')
    if not os.path.exists(model_path):
        print('No trained model found. Train first using train.py')
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        print(f'Cannot read image: {image_path}')
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python predict.py <image_path>')
    else:
        class_id = predict_image(sys.argv[1])
        if class_id is not None:
            print(f'Predicted class: {class_id}')