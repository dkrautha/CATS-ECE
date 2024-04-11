#!/bin/bash

python3 /models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=/models/mymodel/pipeline_file.config \
    --model_dir=/models/mymodel \
    --alsologtostderr \
    --num_train_steps=40000

cp -r /models/mymodel /training

python3 /models/research/object_detection/export_tflite_graph_tf2.py \
    --trained_checkpoint_dir=/models/mymodel/ \
    --output_directory=/training \
    --pipeline_config_path=/models/mymodel/pipeline_file.config
