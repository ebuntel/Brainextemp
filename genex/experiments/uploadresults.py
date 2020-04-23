import boto3

from os import walk
from datetime import datetime

client = boto3.client('s3')

# Create s3 bucket
now = datetime.now()
date_time = now.strftime("%m-%d-medium-datasets")

response = client.create_bucket(
    Bucket=date_time
)

print(response)
#

s3 = boto3.resource('s3')
# Upload files to bucket
mypath = '/Brainextemp/genex/data/results/'
for (dirpath, dirname, filenames) in walk(mypath):
    for direc in dirname:
        for (dirpath2, _, filenames2) in walk(mypath + direc):
            for files in filenames2:
                s3.meta.client.upload_file(dirpath2 + "/" + files, date_time, files)
#