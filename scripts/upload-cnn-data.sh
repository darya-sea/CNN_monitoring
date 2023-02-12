#!/bin/sh

# Upload DATA folder to S3 bucket 

aws s3api create-bucket --bucket cnn-data --create-bucket-configuration LocationConstraint=ap-south-1 || true
aws s3 cp DATA s3://cnn-data --recursive