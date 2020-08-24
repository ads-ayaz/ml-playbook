#!/usr/bin/env python

ATTEMPTS_MAX = 40
DELAY_MIN = 60
DELAY_MAX = 120


import boto3
from botocore.exceptions import ClientError
from time import sleep
import random

client = boto3.client('ec2')


launch_template = {
  'LaunchTemplateId': 'lt-06def67836adb9459',
  # 'LaunchTemplateName': 'string',
  'Version': '2'
}

response='\nUnable to launch instance.'
num_attempts = 0

while num_attempts < ATTEMPTS_MAX :
  try :
    num_attempts += 1
    print("Attempt %i of %i:" % (num_attempts, ATTEMPTS_MAX))

    response = client.run_instances( \
      # InstanceType='m1.large', \
      LaunchTemplate=launch_template, \
      MinCount=1, \
      MaxCount=1 )

    break

  except ClientError as e:
    print("\t" + str(e))
    delay = random.randint(DELAY_MIN, DELAY_MAX)
    print("\tRetry in %i seconds" % delay)
    sleep(delay)

print(response)

