FROM debian:11-slim

ENV DEBIAN_FRONTEND=noninteractive

# SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates=20210119 \
    curl=7.74.0-1.3+deb11u11 \
    unzip=6.0-26+deb11u1 \
    wget=1.21-1+deb11u1 \
    gnupg=2.2.27-2+deb11u2 \
    git=1:2.30.2-1+deb11u2 \
    protobuf-compiler \
    python3-pip \ 
    libgl1 \ 
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/tensorflow/models.git
WORKDIR /models/research
RUN protoc object_detection/protos/*.proto --python_out=.

COPY scripts/modify_tf.py /modify_tf.py
RUN python3 /modify_tf.py

RUN pip install /models/research/
RUN pip install tensorflow==2.8.0
RUN pip install Pillow==9.5.0