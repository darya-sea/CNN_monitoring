import os

CURRENT_DIR = os.path.dirname(__file__)

CNN_FOLDER = os.getenv("CNN_FOLDER", f"{CURRENT_DIR}/CNN")
DATA_FOLDER = os.getenv("DATA_FOLDER", f"{CURRENT_DIR}/DATA")
TRAINING_EPOCHS = 60

POOL_SIZE = 6

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'