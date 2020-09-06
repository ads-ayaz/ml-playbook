#!/bin/bash

# ads-replace-user-data-base64.sh
#
# Replaces all occurences of the token with a base64 encoding of 
# the user data script file inside the given AWS config json file.
#
# Usage:
# ads-replace-user-data-base64.sh <script-file> <config.json>
#
#################################

USER_DATA_SCRIPT=$1
REPLACE_TOKEN=base64_encoded_bash_script

# Use the next line for MacOS
USER_DATA=`base64 $USER_DATA_SCRIPT -b0`
# Or use this line for Ubuntu
#USER_DATA=`base64 $USER_DATA_SCRIPT -w0`

sed -i '' "s|$REPLACE_TOKEN|$USER_DATA|g" $2