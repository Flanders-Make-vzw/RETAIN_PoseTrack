FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y  ca-certificates wget build-essential git libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && \
        apt-get autoremove -y && apt-get clean -y

RUN apt-get update -y && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends python3.9 python3.9-venv python3.9-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 0

# Install dependencies for torch2trt
RUN apt-get update -y && apt-get install -y cmake

# Set Python 3.9 as the default Python version
ENV PYTHONPATH=/usr/bin/python3.9
ENV PATH=/usr/bin/python3.9:$PATH

RUN echo "Python version:" && python3 --version

# Install Python packages globally
RUN python3 -m ensurepip && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install "torch<2.6.0" torchvision "torchaudio<2.6.0" cython pybind11 wheel setuptools 

COPY requirements.txt /tmp/requirements.txt
COPY track /home/PoseTrack/track

# Install pybind11 before installing other requirements
RUN python3 -m pip install pybind11

RUN python3 -m pip install -r /tmp/requirements.txt
RUN python3 -m pip install -e /home/PoseTrack/track/aic_cpp
RUN python3 -m pip install openmim

RUN python3 -m mim install "mmengine>=0.6.0" && \
    python3 -m mim install "mmcv<=2.1.0" && \
    python3 -m mim install "mmpose>=1.1.0" && \
    python3 -m mim install "mmdet>=3.0.0" && \
    python3 -m pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install torch2trt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git /workspace/torch2trt && \
    cd /workspace/torch2trt && \
    python3 setup.py install