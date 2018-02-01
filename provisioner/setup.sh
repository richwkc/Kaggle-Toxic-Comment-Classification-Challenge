#!/bin/bash

set -e

# Installs anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
chmod +x Anaconda3-5.0.1-Linux-x86_64.sh
./Anaconda3-5.0.1-Linux-x86_64.sh -b

# Exports anaconda bin to path
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/anaconda3/bin:$PATH

# Install common libraries
conda install -y pandas scikit-learn scipy gensim

# htop for better monitoring
sudo apt-get install htop

# Download google word2vec model
wget -O "/home/ubuntu/GoogleNews-vectors-negative300.bin.gz" "https://s3-us-west-2.amazonaws.com/toxic-comment-classification/GoogleNews-vectors-negative300.bin.gz"
(cd /home/ubuntu && gunzip "GoogleNews-vectors-negative300.bin.gz")