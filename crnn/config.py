import torch
EPOCHS = 100
BATCH_SIZE = 16
TRAIN_DATA_DIR = 'A:\\aaa\\data\\train'
TEST_DATA_DIR = 'A:\\aaa\\data\\test'
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 80
NUM_WORKERS = 8
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    