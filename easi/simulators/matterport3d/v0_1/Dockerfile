# Matterport3DSimulator with EGL rendering
# Based on: https://github.com/peteanderson80/Matterport3DSimulator
FROM nvidia/cudagl:9.2-devel-ubuntu18.04

# System deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-dev python3-pip python3-setuptools \
    libepoxy-dev libglm-dev libjsoncpp-dev libopencv-dev \
    libosmesa6-dev libglew-dev \
    curl doxygen git \
    && rm -rf /var/lib/apt/lists/*

# CMake 3.12
RUN curl -sSL https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.tar.gz \
    | tar -xz -C /usr/local --strip-components=1

# Python deps
RUN pip3 install --upgrade pip && \
    pip3 install numpy opencv-python networkx

# Build Matterport3DSimulator
WORKDIR /opt/MatterSim
RUN git clone --recursive https://github.com/peteanderson80/Matterport3DSimulator.git . && \
    mkdir build && cd build && \
    cmake -DEGL_RENDERING=ON .. && \
    make -j$(nproc)

ENV PYTHONPATH="/opt/MatterSim/build:/opt/easi"
