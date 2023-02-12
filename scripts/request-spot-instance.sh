#!/bin/sh

# This script to request spot intances

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LAUNCH_TEMPLATE="lt-0f219d87670ad4609"

sed -i '' "s/ACCOUNT_ID/$ACCOUNT_ID/g" spot-fleet-request-config.json
sed -i '' "s/LAUNCH_TEMPLATE/$LAUNCH_TEMPLATE/g" spot-fleet-request-config.json

aws ec2 request-spot-fleet --spot-fleet-request-config file://spot-fleet-request-config.json