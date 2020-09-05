#!/bin/bash

# ads-spot-setup-iam.sh
#
#################################


# Script defaults
ADS_AWS_ACCOUNT_ID=512704425807
ADS_AWS_REGION=ca-central-1
ADS_IAM_ROLE_SPOT_TRAINING=ADS-ML-spot-training-role
ADS_IAM_POLICY_NAME_SPOT_TRAINING=ADS-ML-spot-training-permissions
ADS_IAM_POLICY_FILE_SPOT_TRAINING=ads-ml-spot-training-permissions.json

ADS_PATH_HOME=/home/ubuntu/
ADS_PATH_MOUNT=/ads-ml/
ADS_PATH_CODE=${ADS_PATH_MOUNT}fma-sandbox/
ADS_PATH_CHECKPOINT=${ADS_PATH_MOUNT}training_run/
ADS_PATH_DATA=${ADS_PATH_MOUNT}data/
ADS_PATH_LOG=${ADS_PATH_MOUNT}logs/
ADS_VOLUME_DATASET_NAME=ml-spot-training-data
ADS_VOLUME_DATASET_SIZE=100

# Create a new role for spot ML training
aws iam create-role \
    --role-name ${ADS_IAM_ROLE_SPOT_TRAINING} \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    
# Create policy
aws iam create-policy \
    --policy-name ${ADS_IAM_POLICY_NAME_SPOT_TRAINING}  \
    --policy-document file://${ADS_IAM_POLICY_FILE_SPOT_TRAINING}
    
# Attach the policy to the role
aws iam attach-role-policy \
    --policy-arn arn:aws:iam::${ADS_AWS_ACCOUNT_ID}:policy/${ADS_IAM_POLICY_NAME_SPOT_TRAINING}  \
    --role-name ${ADS_IAM_ROLE_SPOT_TRAINING}
    

