import os

CURRENT_DIR = os.path.dirname(__file__)

CNN_FOLDER = os.getenv("CNN_FOLDER", f"{CURRENT_DIR}/CNN")
POOL_SIZE = 5