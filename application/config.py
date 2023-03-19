import os

CURRENT_DIR = os.path.dirname(__file__)
PARRENT_DIR = os.path.dirname(CURRENT_DIR)

CNN_FOLDER = os.getenv("CNN_FOLDER", f"{PARRENT_DIR}/CNN")
DATA_FOLDER = os.getenv("DATA_FOLDER", f"{PARRENT_DIR}/DATA")
TRAINING_EPOCHS = 60

POOL_SIZE = 6

# AWS params
S3_BUCKET = "cnn-data"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"