import os

CURRENT_DIR = os.path.dirname(__file__)

CNN_FOLDER = os.getenv("CNN_FOLDER", f"{CURRENT_DIR}/CNN")
DATA_FOLDER = os.getenv("CNN_FOLDER", f"{CURRENT_DIR}/DATA")

POOL_SIZE = 6

CSV_FILE = f"{CURRENT_DIR}/Unfitted.csv"