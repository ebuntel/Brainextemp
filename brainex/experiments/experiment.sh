#!/bin/sh
echo "Experiment script beginning."
python3.6 testbed_ucr.py
echo "Experiment finished. Uploading results to S3."
python3.6 uploadresults.py