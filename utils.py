import os
import config


def create_directories():
    os.makedirs(config.MODEL_DIR, exist_ok=True)


def count_classes(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return len(classes)


def get_class_names(data_dir):
    class_names = {}
    for class_id in range(config.NUM_CLASSES):
        class_path = os.path.join(data_dir, f'{class_id:05d}')
        if os.path.exists(class_path):
            class_names[class_id] = f'Class_{class_id:05d}'
    return class_names