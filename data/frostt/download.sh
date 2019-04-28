#!/bin/bash

#wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz
#gunzip < nell-2.tns.gz > nell-2.tns

#wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/amazon/amazon-reviews.tns.gz
#gunzip < amazon-reviews.tns.gz > amazon-reviews.tns

wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/patents/patents.tns.gz
gunzip < patents.tns.gz > patents.tns

wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/reddit-2015/reddit-2015.tns.gz
gunzip < reddit-2015.tns.gz > reddit-2015.tns
