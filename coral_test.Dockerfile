FROM coral:latest

RUN git clone https://github.com/google-coral/pycoral.git

WORKDIR /pycoral

RUN bash examples/install_requirements.sh detect_image.py

CMD ["python3", \
    "examples/detect_image.py", \
    "--model", "model/ssd_mobilenet_v2_catsdogs_quant_edgetpu.tflite", \
    "--labels", "model/labels.txt", \
    "--output", "model/output.jpg", \
    "--input", "model/image.jpg"]