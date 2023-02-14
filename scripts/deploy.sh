#!/bin/sh

aws cloudformation deploy \
    --stack-name cnn-deploy-stack \
    --template-file template.yml \
    --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_IAM \
    --no-fail-on-empty-changeset 
aws s3 sync ../DATA s3://cnn-data