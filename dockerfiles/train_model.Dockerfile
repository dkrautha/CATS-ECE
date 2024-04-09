FROM cats_base:latest

COPY labelmap.pbtxt /labelmap.pbtxt
COPY train.tfrecord /train.tfrecord
COPY valid.tfrecord /val.tfrecord

COPY scripts/download_and_configure_model.py /download_and_configure_model.py 
WORKDIR /models/mymodel
RUN python3 /download_and_configure_model.py

RUN mkdir /training
WORKDIR /training

RUN apt-get update && apt-get install -y --no-install-recommends libgl1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0

CMD [ "python3", \
    "/models/research/object_detection/model_main_tf2.py", \
    "--pipeline_config_path=/models/mymodel/pipeline_file.config", \
    "--model_dir=/models/mymodel", \
    "--alsologtostderr", \
    "--num_train_steps=40000" ]