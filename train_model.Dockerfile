FROM cats_base:latest


COPY scripts/download_model.py /download_model.py 
WORKDIR /models/mymodel
RUN python3 /download_model.py

RUN mkdir /training

CMD ["python3", \
    "/models/research/object_detection/model_train_tf2.py", \
    "--pipeline_config_path=/models/mymodel/pipeline_file.config" \
    "--model_dir=/training", \
    "--alsologtostderr", \
    "--num_train_steps=40000", \
    ]