# Use a CUDA base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04


RUN apt update
RUN apt install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    vim

RUN apt-get autoremove -y 
RUN apt-get clean 
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*=
RUN apt update
RUN apt-get install -y cmake g++ wget

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
