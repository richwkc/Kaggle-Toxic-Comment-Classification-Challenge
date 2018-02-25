#!/bin/bash

set -e

# Set english locale
sudo locale-gen en_CA.UTF-8

# Installs anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
chmod +x Anaconda3-5.0.1-Linux-x86_64.sh
./Anaconda3-5.0.1-Linux-x86_64.sh -b

# Exports anaconda bin to path
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/anaconda3/bin:$PATH

# Install common libraries
conda install -y pandas scikit-learn scipy gensim tensorflow keras

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

wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.6B.50.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.6B.50.txt"
wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.6B.100.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.6B.100.txt"
wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.6B.200.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.6B.200.txt"
wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.6B.300.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.6B.300.txt"
wget -O "/home/ubuntu/state/external-models/glove.6B/w2v.glove.840B.300d.txt" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/w2v-models/glove/w2v.glove.840B.300d.txt"

# htop for better monitoring
sudo apt-get install htop

