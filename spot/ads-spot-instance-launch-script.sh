#!/bin/bash

# ads-spot-instance-launch-script.sh
#
# Description:
# User data script to be run by the spot instance upon first instanciation
# to find the training volume, attach or move it to current availability zone
# and run the training script. Upon completion of training it terminates the 
# fleet that this script belongs to.
#
# Source reference: 
# https://github.com/awslabs/ec2-spot-labs/blob/master/ec2-spot-deep-learning-training/user_data_script.sh
#
# Article reference:
# * Shashank Prasanna | Train Deep Learning Models on GPUs using Amazon EC2 Spot Instances | 27 Mar 2019 
#   https://aws.amazon.com/blogs/machine-learning/train-deep-learning-models-on-gpus-using-amazon-ec2-spot-instances/
#
#################################

# Script defaults
ADS_AWS_REGION=ca-central-1
ADS_CONDA_BIN_ACTIVATE=/home/ubuntu/anaconda3/bin/activate
ADS_CONDA_PROFILE=ads_tf22_p36_spark3
ADS_GIT_USER=ads-USERNAME
ADS_GIT_TOKEN=TOKEN
ADS_GIT_URL=https://$ADS_GIT_USER:$ADS_GIT_TOKEN@github.com/aluance/project.git
ADS_ML_CMD_TRAINING=python\ ./ads-spot-training.py
ADS_PATH_HOME=/home/ubuntu/
ADS_PATH_MOUNT=/ads-ml/
ADS_PATH_CODE=${ADS_PATH_HOME}project/
ADS_PATH_DATA=${ADS_PATH_MOUNT}data/
ADS_PATH_DEBUG=${ADS_PATH_MOUNT}debug/
ADS_VOLUME_DATASET_NAME=ml-spot-training-data
ADS_VOLUME_LABEL=adsvol-spot-training

echo "Started ads-spot-instance-launch-script.sh ..."

# Get instance ID, Instance availability zone, volume ID and volume availability zone
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)

echo "  Instance ID: ${INSTANCE_ID}"
echo "  Instance AZ: ${INSTANCE_AZ}"

VOLUME_ID=$(aws ec2 describe-volumes --region $ADS_AWS_REGION --filter "Name=tag:Name,Values=$ADS_VOLUME_DATASET_NAME" --query "Volumes[].VolumeId" --output text)
VOLUME_AZ=$(aws ec2 describe-volumes --region $ADS_AWS_REGION --filter "Name=tag:Name,Values=$ADS_VOLUME_DATASET_NAME" --query "Volumes[].AvailabilityZone" --output text)

# Proceed if Volume Id is not null or unset
if [ $VOLUME_ID ]; then
	# Check if the Volume AZ and the instance AZ are same or different.
	# If they are different, create a snapshot and then create a new volume in the instance's AZ.
	if [ $VOLUME_AZ != $INSTANCE_AZ ]; then
        echo "Volume and instance are in different availability zones. Recreating volume in ${INSTANCE_AZ}."

        echo "Creating snapshot..."
		SNAPSHOT_ID=$(aws ec2 create-snapshot \
    		--region $ADS_AWS_REGION \
    		--volume-id $VOLUME_ID \
    		--description "`date +"%D %T"`" \
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

    echo "Attaching volume to this instance ..."

    # Attach the volume to this instance
    aws ec2 attach-volume \
        --region $ADS_AWS_REGION \
        --volume-id $VOLUME_ID \
        --instance-id $INSTANCE_ID \
        --device /dev/sdf
    sleep 10
    
    # Create the mount path folder if it doesn't exist
    if ! [ -d ${ADS_PATH_MOUNT} ]; then
        echo "Creating the ${ADS_PATH_MOUNT} mount folder ..."
        mkdir --parents ${ADS_PATH_MOUNT}
    fi

    # Mount the device to the mount path and create paths as needed
    echo "Attempting to mount volume with label ${ADS_VOLUME_LABEL} at ${ADS_PATH_MOUNT} ..."
    sleep 5
    sudo mount --label ${ADS_VOLUME_LABEL} ${ADS_PATH_MOUNT}
    sudo chown -R ubuntu: ${ADS_PATH_MOUNT}
    
    # Get training code
    echo "Cloning code from ${ADS_GIT_URL}..."
    cd $ADS_PATH_HOME
    git clone $ADS_GIT_URL
    chown -R ubuntu: $ADS_PATH_CODE

	# Initiate training
	echo "Starting training from ${ADS_PATH_CODE}."
    cd $ADS_PATH_CODE
	sudo -H -u ubuntu bash -c "source ${ADS_CONDA_BIN_ACTIVATE} ${ADS_CONDA_PROFILE}; ${ADS_ML_CMD_TRAINING}"
fi

# Copy the cloud-init output log file to the volume for debugging 
if ! [ -d ${ADS_PATH_DEBUG} ]; then
    mkdir --parents ${ADS_PATH_DEBUG}
fi
cp /var/log/cloud-init-output.log "${ADS_PATH_DEBUG}cloud-init-output_`date +"%Y%m%d%H%M%S"`.log"

# After training, clean up by cancelling spot requests and terminating itself
SPOT_FLEET_REQUEST_ID=$(aws ec2 describe-spot-instance-requests \
    --region $ADS_AWS_REGION \
    --filter "Name=instance-id,Values='$INSTANCE_ID'" \
    --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" \
    --output text)

echo "Shutting down spot fleet with ID: ${SPOT_FLEET_REQUEST_ID} ..."

aws ec2 cancel-spot-fleet-requests \
    --region $ADS_AWS_REGION \
    --spot-fleet-request-ids $SPOT_FLEET_REQUEST_ID \
    --terminate-instances
    