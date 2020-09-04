#!/bin/bash

# Set script defaults
ADS_VOLUME_DATASET_NAME=ads-data
AWS_REGION=ca-central-1

# Get instance ID, Instance availability zone, volume ID and volume availability zone
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)

VOLUME_ID=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=$ADS_VOLUME_DATASET_NAME" --query "Volumes[].VolumeId" --output text)
VOLUME_AZ=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=$ADS_VOLUME_DATASET_NAME" --query "Volumes[].AvailabilityZone" --output text)

