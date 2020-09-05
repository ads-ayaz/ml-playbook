#!/usr/bin/env python

ATTEMPTS_MAX = None #400
ATTEMPTS_DURATION = None #6
DELAY_MIN = 60
DELAY_MAX = 120


import boto3
from botocore.exceptions import ClientError
import random
import time
from time import sleep

# Set maximum time for attempts. Default is 1 hour
t_max = time.time() + 3600
if ATTEMPTS_DURATION is not None :
  t_max = time.time() + (3600 * ATTEMPTS_DURATION)

# Set maximum number of attempts. Default is 10
attempts_max = 10
if ATTEMPTS_MAX is not None :
  attempts_max = ATTEMPTS_MAX


client = boto3.client('ec2')

launch_template = {
  'LaunchTemplateId': 'lt-06def67836adb9459',
  # 'LaunchTemplateName': 'string',
  'Version': '2'
}

placement = {
  'AvailabilityZone': 'ca-central-1b'
}

response='\nUnable to launch instance.'
num_attempts = 0

while (num_attempts < attempts_max) and (time.time() < t_max) :
  try :
    num_attempts += 1
    print("Attempt %i of %i:" % (num_attempts, attempts_max))

    response = client.run_instances( \
      InstanceType='g4dn.8xlarge', \
      LaunchTemplate=launch_template, \
      # Placement=placement, \
      MinCount=1, \
      MaxCount=1 )

    break

  except ClientError as e:
    print("\t" + str(e))
    if num_attempts < attempts_max :
      delay = random.randint(DELAY_MIN, DELAY_MAX)
      print("\tRetry in %i seconds" % delay)
      sleep(delay)

print(response)

