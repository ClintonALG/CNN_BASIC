import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config


class GTSRBDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def load_from_csv(csv_path, img_dir, img_size):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['Path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(row['ClassId'])

    return np.array(images), np.array(labels)


def get_dataloaders():
    print('Loading training data...')
    X, y = load_from_csv(config.TRAIN_CSV, config.DATA_DIR, config.IMG_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = GTSRBDataset(X_train, y_train)
    val_dataset = GTSRBDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f'Train samples: {len(X_train)}, Val samples: {len(X_val)}')

    return train_loader, val_loader


def get_test_loader():
    print('Loading test data...')
    X_test, y_test = load_from_csv(config.TEST_CSV, config.DATA_DIR, config.IMG_SIZE)

    test_dataset = GTSRBDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f'Test samples: {len(X_test)}')
    return test_loader