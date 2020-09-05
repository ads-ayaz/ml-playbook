#!/bin/bash

# ads-spot-setup.sh
#
# Description:
# User data script to set up a volume for spot training. It will create the volume (if needed),
# mount it and create folders on the volume, and sync the training data from S3 volume.
#
# Assumptions:
# This is being run from an EC2 instance that already has the aws CLI configured and necessary IAM permissions
#
#################################


# Script defaults
ADS_AWS_REGION=ca-central-1
#ADS_CONDA_BIN_ACTIVATE=/home/ubuntu/anaconda3/bin/activate
#ADS_CONDA_PROFILE=ads_tf22_p36_spark3
ADS_GIT_URL=https://github.com/ads-ayaz/fma-sandbox.git
#ADS_ML_CMD_TRAINING=python ./ads-training-spot.py
ADS_PATH_HOME=/home/ubuntu/
ADS_PATH_MOUNT=/ads-ml/
ADS_PATH_CODE=${ADS_PATH_MOUNT}fma-sandbox/
ADS_PATH_CHECKPOINT=${ADS_PATH_MOUNT}training_run/
ADS_PATH_DATA=${ADS_PATH_MOUNT}data/
ADS_PATH_LOG=${ADS_PATH_MOUNT}logs/
ADS_VOLUME_DATASET_NAME=ml-spot-training-data
ADS_VOLUME_DATASET_SIZE=100



# Determine the current instance ID and availability zone
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Find the training dataset volume and its availability zone
VOLUME_ID=$(aws ec2 describe-volumes --region ${ADS_AWS_REGION} --filter "Name=tag:Name,Values=${ADS_VOLUME_DATASET_NAME}" --query "Volumes[].VolumeId" --output text)
VOLUME_AZ=$(aws ec2 describe-volumes --region ${ADS_AWS_REGION} --filter "Name=tag:Name,Values=${ADS_VOLUME_DATASET_NAME}" --query "Volumes[].AvailabilityZone" --output text)

# If no volume is found then create and initialize a volume
if [ ! $VOLUME_ID ]; then

    # Create the new volume and get its ID; wait for it to become available before proceeding.
    VOLUME_ID=$(aws ec2 create-volume \
        --size ${ADS_VOLUME_DATASET_SIZE} \
        --region ${ADS_AWS_REGION} \
        --availability-zone ${INSTANCE_AZ} \
        --volume-type gp2 \
        --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=${ADS_VOLUME_DATASET_NAME}}]' \
		--query VolumeId \
		--output text)
    aws ec2 wait volume-available --region ${ADS_AWS_REGION} --volume-id ${VOLUME_ID}
fi

# If the volume AZ and this instance AZ are different, then create a snapshot 
# and create a new volume in this instance's AZ.
if [ $VOLUME_AZ != $INSTANCE_AZ ]; then
	SNAPSHOT_ID=$(aws ec2 create-snapshot \
		--region $ADS_AWS_REGION \
		--volume-id $VOLUME_ID \
		--description "`date +"%D %T"`" \
		--tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=$ADS_VOLUME_DATASET_NAME}]' \
		--query SnapshotId 
		--output text)
		
	# Wait for the snapshot to be completed, then delete the old volume
	aws ec2 wait --region $ADS_AWS_REGION snapshot-completed --snapshot-ids $SNAPSHOT_ID
	aws ec2 --region $ADS_AWS_REGION  delete-volume --volume-id $VOLUME_ID
	
	# Create a new volume in the same availability zone from the snapshot
	# and wait until it becomes available
	VOLUME_ID=$(aws ec2 create-volume \
		--region $ADS_AWS_REGION \
		--availability-zone $INSTANCE_AZ \
		--snapshot-id $SNAPSHOT_ID \
		--volume-type gp2 \
		--tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=$ADS_VOLUME_DATASET_NAME}]' \
		--query VolumeId \
		--output text)
    aws ec2 wait volume-available --region $ADS_AWS_REGION --volume-id $VOLUME_ID
fi

# Now that we have a volume in our AZ, attach it and get the device name 
DEVICE_NAME=$(aws ec2 attach-volume \
    --volume-id vol-${VOLUME_ID} \
    --instance-id i-<your_instance_id> \
    --device /dev/sdf \
    --query Device
    --output text)

# Create the mount path folder if it does not exist
if [ ! [ d ${ADS_PATH_MOUNT} ]]; then
    sudo mkdir --parents ${ADS_PATH_MOUNT}
fi

# Check that the volume has a filesystem; otherwise create one
DEVICE_FILESYSTEM=$(sudo file -s ${DEVICE_NAME})
if [[ "${DEVICE_FILESYSTEM}" == *": data"* ]]; then
    sudo mkfs -t xfs ${DEVICE_NAME}

    # Mount the device and set owner to ubuntu then un-mount
    sudo mount ${DEVICE_NAME} ${ADS_PATH_MOUNT}
    sudo chown -R ubuntu: ${ADS_PATH_MOUNT}
    sudo umount ${ADS_PATH_MOUNT}
fi

# Mount the device to the mount path and create paths as needed
sudo mount ${DEVICE_NAME} ${ADS_PATH_MOUNT}

if [! [d ${ADS_PATH_CHECKPOINT} ]]; then
    sudo mkdir --parents ${ADS_PATH_CHECKPOINT}
    sudo chown -R ubuntu: ${ADS_PATH_CHECKPOINT}
fi
if [! [d ${ADS_PATH_DATA} ]]; then
    sudo mkdir --parents ${ADS_PATH_DATA}
    sudo chown -R ubuntu: ${ADS_PATH_DATA}
fi
if [! [d ${ADS_PATH_LOG} ]]; then
    sudo mkdir --parents ${ADS_PATH_LOG}
    sudo chown -R ubuntu: ${ADS_PATH_LOG}
fi

# Sync data from an AWS S3 bucket as ubuntu user. (This will have no effect if the data is already synced)
sudo -H -u ubuntu bash -c "aws s3 sync ${ADS_S3_PATH_DATA_PROCESSED} ${ADS_PATH_DATA}"

# Unmount the volume before ending
sudo umount ${ADS_PATH_MOUNT}
