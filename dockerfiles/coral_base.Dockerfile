FROM cats_base:latest

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends \
    libedgetpu1-std \
    python3-pycoral \
    python3-opencv \
    edgetpu-compiler \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*