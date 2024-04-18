import re  # noqa: INP001, D100
import subprocess
import tarfile
from pathlib import Path

train_record_filename = "/train.tfrecord"
validate_record_filename = "/val.tfrecord"
label_map_filename = "/labelmap.pbtxt"

# model_name = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
model_name = "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03"

link = ""

input_pipeline_name = f"{model_name}.config"
pretrained_checkpoint = f"{model_name}.tar.gz"

num_steps = 40000
batch_size = 16

output_pipeline_name = "/models/mymodel/pipeline_file.config"
fine_tune_checkpoint = f"/models/mymodel/{model_name}/checkpoint/ckpt-0"


def download_model() -> None:
    # download_tar = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{pretrained_checkpoint}"
    download_tar = f"http://download.tensorflow.org/models/object_detection/{pretrained_checkpoint}"
    subprocess.run(["/usr/bin/wget", download_tar], check=True)  # noqa: S603
    tar = tarfile.open(pretrained_checkpoint)
    tar.extractall()  # noqa: S202
    tar.close()


def download_config_file() -> None:
    download_config = f"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/{input_pipeline_name}"
    subprocess.run(["/usr/bin/wget", download_config], check=True)  # noqa: S603


def get_num_classes() -> int:
    from object_detection.utils import label_map_util

    label_map = label_map_util.load_labelmap(label_map_filename)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=90,
        use_display_name=True,
    )
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


def modify_config_file() -> None:
    num_classes = get_num_classes()
    with Path(input_pipeline_name).open("r") as f:
        s = f.read()
    with Path(output_pipeline_name).open("w") as f:
        # Set fine_tune_checkpoint path
        s = re.sub(
            'fine_tune_checkpoint: ".*?"',
            f'fine_tune_checkpoint: "{fine_tune_checkpoint}"',
            s,
        )

        # Set tfrecord files for train and test datasets
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
            f'input_path: "{train_record_filename}"',
            s,
        )
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
            f'input_path: "{validate_record_filename}"',
            s,
        )

        # Set label_map_path
        s = re.sub(
            'label_map_path: ".*?"',
            f'label_map_path: "{label_map_filename}"',
            s,
        )

        s = re.sub("batch_size: [0-9]+", f"batch_size: {batch_size}", s)
        s = re.sub("num_steps: [0-9]+", f"num_steps: {num_steps}", s)
        s = re.sub("num_classes: [0-9]+", f"num_classes: {num_classes}", s)
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"',
            'fine_tune_checkpoint_type: "detection"',
            s,
        )

        # If using ssd-mobilenet-v2, reduce learning rate
        # (because it's too high in the default config file)
        s = re.sub("learning_rate_base: .8", "learning_rate_base: .08", s)
        s = re.sub("warmup_learning_rate: 0.13333", "warmup_learning_rate: .026666", s)

        f.write(s)


def main() -> None:
    download_model()
    download_config_file()
    modify_config_file()


if __name__ == "__main__":
    main()

__all__ = []
