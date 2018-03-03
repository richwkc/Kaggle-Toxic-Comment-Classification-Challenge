#!/bin/bash

set -e

# Set english locale
sudo locale-gen en_CA.UTF-8

# Install common libraries
(cd /home/ubuntu && source /home/ubuntu/anaconda3/bin//activate tensorflow_p36 && pip install gensim sklearn)

# Download data-sets and w2v model
mkdir "/home/ubuntu/state"
mkdir "/home/ubuntu/state/data"
mkdir "/home/ubuntu/state/data/preprocessed-train-test"
mkdir "/home/ubuntu/state/external-models"
mkdir "/home/ubuntu/state/external-models/glove.6B"

wget -O "/home/ubuntu/state/data/preprocessed-train-test/all.csv" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/data/all.csv"
wget -O "/home/ubuntu/state/data/preprocessed-train-test/train.csv" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/data/train.csv"
wget -O "/home/ubuntu/state/data/preprocessed-train-test/test.csv" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/data/test.csv"
wget -O "/home/ubuntu/state/data/preprocessed-train-test/contest-test.csv" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/data/contest-test.csv"

#wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.6B.50.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.6B.50.txt"
#wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.6B.100.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.6B.100.txt"
#wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.6B.200.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.6B.200.txt"
#wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.6B.300.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.6B.300.txt"
#wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.840B.300d.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.840B.300d.txt"
wget -O "/home/ubuntu/state/external-models/glove.6B/webcrawl.bin" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/webcrawl.bin"
wget -O "/home/ubuntu/state/external-models/glove.6B/webcrawl.bin.vectors.npy" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/webcrawl.bin.vectors.npy"
