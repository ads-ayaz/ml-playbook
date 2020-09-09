#!/bin/bash

# ads-spot-setup-iam.sh
#
# Description:
# Script creates necessary IAM roles for automated ML spot fleet training.
#
# Assumptions:
# This is being run from an EC2 instance that already has the aws CLI configured.
#
#################################


# Script defaults
ADS_AWS_ACCOUNT_ID=512704425807
ADS_AWS_REGION=ca-central-1
ADS_IAM_ROLE_SPOT_FLEET=ADS-ML-spot-fleet-role
ADS_IAM_ROLE_SPOT_TRAINING=ADS-ML-spot-training-role
ADS_IAM_POLICY_NAME_SPOT_TRAINING=ADS-ML-spot-training-permissions
ADS_IAM_POLICY_FILE_SPOT_TRAINING=ads-spot-training-permissions.json


ADS_PATH_HOME=/home/ubuntu/
ADS_PATH_MOUNT=/ads-ml/
ADS_PATH_CODE=${ADS_PATH_MOUNT}fma-sandbox/
ADS_PATH_CHECKPOINT=${ADS_PATH_MOUNT}training_run/
ADS_PATH_DATA=${ADS_PATH_MOUNT}data/
ADS_PATH_LOG=${ADS_PATH_MOUNT}logs/
ADS_VOLUME_DATASET_NAME=ml-spot-training-data
ADS_VOLUME_DATASET_SIZE=100

# Create a new role for spot ML training instance
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
    

# Create a new role for spot ML training fleet
aws iam create-role \
     --role-name ${ADS_IAM_ROLE_SPOT_FLEET} \
     --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"spotfleet.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

# Attach AmazonEC2SpotFleetTaggingRole to the fleet policy
aws iam attach-role-policy \
     --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole --role-name ${ADS_IAM_ROLE_SPOT_FLEET}
     