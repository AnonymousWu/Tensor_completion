#!/bin/bash

wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/amazon/amazon-reviews.tns.gz
gunzip < amazon-reviews.tns.gz > amazon-reviews.tns
split -l 300000000 amazon-reviews.tns
mv xaa xaa.tns
