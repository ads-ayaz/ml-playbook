#!/bin/bash

# ads-spot-setup.sh
#
# Description:
# Script to set up a volume for spot training. It will check that the volume is created,
# mount it and create folders on the volume, and sync the training data from S3 volume.
#
# Assumptions:
# * This is being run from an EC2 instance that already has the aws CLI configured and necessary IAM permissions.
# * Volume must already have been created and filesystem exists.
#
#################################


# Script defaults
ADS_AWS_REGION=ca-central-1
ADS_GIT_URL=https://github.com/ads-ayaz/fma-sandbox.git
ADS_PATH_HOME=/home/ubuntu/
ADS_PATH_MOUNT=/ads-ml/
ADS_PATH_CODE=${ADS_PATH_MOUNT}fma-sandbox/
ADS_PATH_CHECKPOINT=${ADS_PATH_MOUNT}training_run/
ADS_PATH_DATA=${ADS_PATH_MOUNT}data/
ADS_PATH_LOG=${ADS_PATH_MOUNT}logs/
ADS_S3_PATH_DATA_PROCESSED=s3://fma-sandbox-ml-tempdata/data-processed/data_large/
ADS_VOLUME_DATASET_NAME=ml-spot-training-data
ADS_VOLUME_DATASET_SIZE=100
ADS_VOLUME_LABEL=adsvol-data




# Determine the current instance ID and availability zone
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Find the training dataset volume and its availability zone
VOLUME_ID=$(aws ec2 describe-volumes --region ${ADS_AWS_REGION} --filter "Name=tag:Name,Values=${ADS_VOLUME_DATASET_NAME}" --query "Volumes[].VolumeId" --output text)
VOLUME_AZ=$(aws ec2 describe-volumes --region ${ADS_AWS_REGION} --filter "Name=tag:Name,Values=${ADS_VOLUME_DATASET_NAME}" --query "Volumes[].AvailabilityZone" --output text)

# If no volume is found then tell user they need to create / label the volume
if ! [ $VOLUME_ID ]; then
    echo "ERROR: Unable to find volume named ${ADS_VOLUME_DATASET_NAME} in ${ADS_AWS_REGION} region."
    exit 1
    
    # Create the new volume and get its ID; wait for it to become available before proceeding.
#     VOLUME_ID=$(aws ec2 create-volume \
#         --size ${ADS_VOLUME_DATASET_SIZE} \
#         --region ${ADS_AWS_REGION} \
#         --availability-zone ${INSTANCE_AZ} \
#         --volume-type gp2 \
#         --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=${ADS_VOLUME_DATASET_NAME}}]" \
# 		--query VolumeId \
# 		--output text)
#     aws ec2 wait volume-available --region ${ADS_AWS_REGION} --volume-id ${VOLUME_ID}
fi

echo "Found volume ${ADS_VOLUME_DATASET_NAME} with ID ${VOLUME_ID} in ${VOLUME_AZ}."

# If the volume AZ and this instance AZ are different, then create a snapshot 
# and create a new volume in this instance's AZ.
if ! [ $VOLUME_AZ == $INSTANCE_AZ ]; then
    echo "Volume and instance are in different availability zones. Recreating volume in ${INSTANCE_AZ}."
    
    echo "Creating snapshot..."
	SNAPSHOT_ID=$(aws ec2 create-snapshot \
		--region $ADS_AWS_REGION \
		--volume-id $VOLUME_ID \
		--description "Created `date +"%D %T"` UTC" \
		--tag-specifications "ResourceType=snapshot,Tags=[{Key=Name,Value=$ADS_VOLUME_DATASET_NAME}]" \
		--query SnapshotId \
		--output text)
		
	# Wait for the snapshot to be completed, then delete the old volume
	while [ "${exit_status}" != "0" ]
    do
        SNAPSHOT_STATE="$(aws ec2 describe-snapshots --filters Name=snapshot-id,Values=${SNAPSHOT_ID} --query 'Snapshots[0].State')"
        SNAPSHOT_PROGRESS="$(aws ec2 describe-snapshots --filters Name=snapshot-id,Values=${SNAPSHOT_ID} --query 'Snapshots[0].Progress')"
        echo "### Snapshot id ${SNAPSHOT_ID} creation: state is ${SNAPSHOT_STATE}, ${SNAPSHOT_PROGRESS}..."
    
    	aws ec2 wait --region $ADS_AWS_REGION snapshot-completed --snapshot-ids $SNAPSHOT_ID
        exit_status="$?"
    done
	aws ec2 --region $ADS_AWS_REGION  delete-volume --volume-id $VOLUME_ID
	
	echo "Snapshot created. Old volume deleted."
	echo "Creating a new volume from the image..."
	
	# Create a new volume in the same availability zone from the snapshot
	# and wait until it becomes available
	VOLUME_ID=$(aws ec2 create-volume \
		--region $ADS_AWS_REGION \
		--availability-zone $INSTANCE_AZ \
		--snapshot-id $SNAPSHOT_ID \
		--volume-type gp2 \
		--tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=$ADS_VOLUME_DATASET_NAME}]" \
		--query VolumeId \
		--output text)
    aws ec2 wait volume-available --region $ADS_AWS_REGION --volume-id $VOLUME_ID
    
    echo "New volume with ID ${VOLUME_ID} created in ${INSTANCE_AZ}."
fi

# Now that we have a volume in our AZ, attach it and get the device name 
echo "Attaching volume ID ${VOLUME_ID} to this instance."
aws ec2 attach-volume \
    --region $ADS_AWS_REGION \
    --volume-id ${VOLUME_ID} \
    --instance-id ${INSTANCE_ID} \
    --device /dev/sdf
sleep 10

# Create the mount path folder if it does not exist
if ! [ -d ${ADS_PATH_MOUNT} ]; then
    echo "Creating ${ADS_PATH_MOUNT} folder."
    sudo mkdir --parents ${ADS_PATH_MOUNT}
fi

# Check that the volume has a filesystem; otherwise create one
# DEVICE_FILESYSTEM=$(sudo file -s ${DEVICE_NAME})
# if [[ "${DEVICE_FILESYSTEM}" == *": data"* ]]; then
#     sudo mkfs -t xfs -L ${ADS_VOLUME_LABEL} ${DEVICE_NAME}

#     # Mount the device and set owner to ubuntu then un-mount
#     sudo mount --label ${ADS_VOLUME_LABEL} ${ADS_PATH_MOUNT}
#     sudo chown -R ubuntu: ${ADS_PATH_MOUNT}
#     sudo umount ${ADS_PATH_MOUNT}
# fi

# Mount the device to the mount path and create paths as needed
echo "Attempting to mount volume with label ${ADS_VOLUME_LABEL} at ${ADS_PATH_MOUNT} ..."
sleep 5
sudo mount --label ${ADS_VOLUME_LABEL} ${ADS_PATH_MOUNT}
sudo chown -R ubuntu: ${ADS_PATH_MOUNT}

if ! [ -d ${ADS_PATH_CHECKPOINT} ]; then
    sudo mkdir --parents ${ADS_PATH_CHECKPOINT}
    sudo chown -R ubuntu: ${ADS_PATH_CHECKPOINT}
    echo "Created ${ADS_PATH_CHECKPOINT} ."
fi
if ! [ -d ${ADS_PATH_DATA} ]; then
    sudo mkdir --parents ${ADS_PATH_DATA}
    sudo chown -R ubuntu: ${ADS_PATH_DATA}
    echo "Created ${ADS_PATH_DATA} ."
fi
if ! [ -d ${ADS_PATH_LOG} ]; then
    sudo mkdir --parents ${ADS_PATH_LOG}
    sudo chown -R ubuntu: ${ADS_PATH_LOG}
    echo "Created ${ADS_PATH_LOG} ."
fi

# Sync data from an AWS S3 bucket as ubuntu user. (This will have no effect if the data is already synced)
echo "Synchronizing from ${ADS_S3_PATH_DATA_PROCESSED} to ${ADS_PATH_DATA} ..."
sudo -H -u ubuntu bash -c "aws s3 sync ${ADS_S3_PATH_DATA_PROCESSED} ${ADS_PATH_DATA}"

# Unmount the volume before ending
echo "Unmounting ${ADS_PATH_MOUNT} ..."
sudo umount ${ADS_PATH_MOUNT}

# Detach the volume from this instance
echo "Detaching volume ID ${VOLUME_ID} from this instance."
aws ec2 detach-volume \
    --volume-id ${VOLUME_ID}

echo "All done!"
