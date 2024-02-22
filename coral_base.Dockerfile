FROM debian:11-slim

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates=20210119 \
    curl=7.74.0-1.3+deb11u11 \
    unzip=6.0-26+deb11u1 \
    wget=1.21-1+deb11u1 \
    gnupg=2.2.27-2+deb11u2 \
    git=1:2.30.2-1+deb11u2 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends \
    libedgetpu1-std \
    python3-pycoral \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
