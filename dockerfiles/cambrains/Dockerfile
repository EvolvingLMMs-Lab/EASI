ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE} AS base

# Set work directory
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y vim tmux zip unzip wget git git-lfs build-essential libibverbs-dev ca-certificates \
    curl iproute2 libsm6 libxext6 kmod ccache libnuma-dev libssl-dev flex bison libgtk-3-dev libpango1.0-dev \
    libsoup2.4-dev libnice-dev libopus-dev libvpx-dev libx264-dev libsrtp2-dev libglib2.0-dev libdrm-dev\
    && apt-get clean && rm -rf /var/lib/apt/lists/* && git lfs install

# Install VLMEvalkit
COPY . .
RUN pip install -e .

# Install Cambrian-S
WORKDIR /opt
RUN git clone https://github.com/cambrian-mllm/cambrian-s.git && \
    cd cambrian-s && \
    git checkout 383f321 && \
    pip install -e .

WORKDIR /app
