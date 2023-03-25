import os

CURRENT_DIR = os.path.dirname(__file__)
PARRENT_DIR = os.path.dirname(CURRENT_DIR)

CNN_FOLDER = os.getenv("CNN_FOLDER", f"{PARRENT_DIR}/CNN")
DATA_FOLDER = os.getenv("DATA_FOLDER", f"{PARRENT_DIR}/DATA")
TRAINING_EPOCHS = 50

POOL_SIZE = 6

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# AWS params
S3_BUCKET = os.getenv("S3_BUCKET", "cnn-data")
EC2_MAX_PRICE = os.getenv("EC2_MAX_PRICE", "0.7")
EC2_AMI_ID = os.getenv("EC2_AMI_ID", "ami-07600570b75d0d064")
EC2_INSTANCE_TYPES = [
    {
        "InstanceType": "g4dn.xlarge"
    },
    {
        "InstanceType": "g4dn.2xlarge"
    },
    {
        "InstanceType": "g5g.xlarge"
    },
    {
        "InstanceType": "g5g.2xlarge"
    }
]
