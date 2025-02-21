#!/bin/bash
set -e

# Initialize conda
eval "$(/opt/miniconda/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# Create and activate the conda environment
conda create -n aic24 python=3.9 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda activate aic24

# Install the required packages
pip install -r /tmp/requirements.txt
pip install -e /home/PoseTrack/track
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv<=2.1.0"
mim install "mmpose>=1.1.0"
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
