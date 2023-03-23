import os

CURRENT_DIR = os.path.dirname(__file__)
PARRENT_DIR = os.path.dirname(CURRENT_DIR)

CNN_FOLDER = os.getenv("CNN_FOLDER", f"{PARRENT_DIR}/CNN")
DATA_FOLDER = os.getenv("DATA_FOLDER", f"{PARRENT_DIR}/DATA")
TRAINING_EPOCHS = 60

POOL_SIZE = 6

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# AWS params
S3_BUCKET = "cnn-data"
EC2_MAX_PRICE = "0.07"
EC2_INSTANCE_TYPES = [
    {
        "InstanceType": "t3a.xlarge"
    },
    {
        "InstanceType": "t3.xlarge"
    },
    {
        "InstanceType": "c5a.xlarge"
    },
    {
        "InstanceType": "c6i.xlarge"
    },
    {
        "InstanceType": "m5.xlarge"
    },
    {
        "InstanceType": "c4.xlarge"
    },
    {
        "InstanceType": "r4.xlarge"
    },
    {
        "InstanceType": "r5n.xlarge"
    },
    {
        "InstanceType": "m6i.xlarge"
    },
    {
        "InstanceType": "t2.xlarge"
    },
    {
        "InstanceType": "r5.xlarge"
    },
    {
        "InstanceType": "c6a.xlarge"
    },
    {
        "InstanceType": "r6i.xlarge"
    },
    {
        "InstanceType": "m6a.xlarge"
    },
    {
        "InstanceType": "r6a.xlarge"
    },
    {
        "InstanceType": "c5n.xlarge"
    },
    {
        "InstanceType": "m5a.xlarge"
    },
    {
        "InstanceType": "c5.xlarge"
    }
]

# EC2_INSTANCE_TYPES = [
#     {
#         "InstanceType": "g4dn.xlarge"
#     }
# ]
  
# #g4dn.2xlarge