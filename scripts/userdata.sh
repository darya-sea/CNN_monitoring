#!/bin/bash

# Bootstrap script for AWS EC2 instance. 

PERSISTENT_VOLUME_ID="vol-06278d1ce69990e90"
REGION="ap-south-1"

INSTANCE_ID=$(curl http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 attach-volume --volume-id $PERSISTENT_VOLUME_ID --device /dev/xvdf --instance-id $INSTANCE_ID --region $REGION
DATA_STATE="unknown"
until [ "$DATA_STATE" == "attached" ]; do
    DATA_STATE=$(aws ec2 describe-volumes \
    --region $REGION \
    --filters \
        Name=attachment.instance-id,Values=$INSTANCE_ID \
        Name=attachment.device,Values=/dev/xvdf \
    --query Volumes[].Attachments[].State \
    --output text)

    sleep 5
done

mount /dev/xvdf /mnt
cd /mnt
git clone https://github.com/darya-sea/CNN_monitoring.git
cd CNN_monitoring
pip3 install -r requirements.txt -t .
aws s3 sync s3://cnn-data DATA
python3 main.py train
aws s3 sync DATA s3://cnn-data
