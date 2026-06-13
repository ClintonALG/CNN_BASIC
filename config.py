import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')

TRAIN_CSV = os.path.join(DATA_DIR, 'Train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'Test.csv')
TRAIN_DIR = os.path.join(DATA_DIR, 'Train')
TEST_DIR = os.path.join(DATA_DIR, 'Test')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 43

DEVICE = 'cuda'