ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE} AS base

# Set work directory
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y vim tmux zip unzip wget git git-lfs build-essential libibverbs-dev ca-certificates \
    curl iproute2 libsm6 libxext6 kmod ccache libnuma-dev libssl-dev flex bison libgtk-3-dev libpango1.0-dev \
    libsoup2.4-dev libnice-dev libopus-dev libvpx-dev libx264-dev libsrtp2-dev libglib2.0-dev libdrm-dev\
    && apt-get clean && rm -rf /var/lib/apt/lists/* && git lfs install

# Install VLMEvalkit
COPY . .
RUN pip install -e .

# Install vlm3r
WORKDIR /opt
RUN git clone --recursive https://github.com/VITA-Group/VLM-3R.git && \
    cd VLM-3R && \
    pip install -e .

# Install CUT3R (build from source)
RUN cd VLM-3R/CUT3R && \
    pip install -r requirements.txt && \
    cd src/croco/models/curope/ && \
    python setup.py build_ext --inplace

# Download pre-trained CUT3R model weights
RUN hf download lmms-lab-si/third-party-models --include "VLM3R/cut3r_512_dpt_4_64.pth"  --local-dir ./VLM-3R/CUT3R/src && \
    mv ./VLM-3R/CUT3R/src/VLM3R/cut3r_512_dpt_4_64.pth ./VLM-3R/CUT3R/src && \
    rm -rf ./VLM-3R/CUT3R/src/VLM3R

# Install some dependencies
RUN pip install peft==0.10.0 flash-attn==2.7.4.post1 transformers==4.40.0
WORKDIR /app
