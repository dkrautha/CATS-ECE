FROM coral:latest

RUN git clone https://github.com/google-coral/pycoral.git
WORKDIR /pycoral 
RUN bash examples/install_requirements.sh classify_image.py

CMD ["python3", \
    "examples/classify_image.py", \
    "--model", "test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite", \
    "--labels", "test_data/inat_bird_labels.txt", \
    "--input", "test_data/parrot.jpg"]