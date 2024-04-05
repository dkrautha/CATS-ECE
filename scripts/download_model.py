import re  # noqa: INP001, D100
import subprocess
import tarfile
from pathlib import Path

train_record_fname = "/train.tfrecord"
val_record_fname = "/val.tfrecord"
label_map_pbtxt_fname = "/labelmap.pbtxt"


def download_model() -> None:
    # Download pre-trained model weights
    pretrained_checkpoint = "ssd_mobilenet_v2_320x320_coco17_tpu-8.config"
    download_tar = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{pretrained_checkpoint}"
    subprocess.run(["/usr/bin/wget", download_tar], check=True)  # noqa: S603
    tar = tarfile.open(pretrained_checkpoint)
    tar.extractall()  # noqa: S202
    tar.close()


base_pipeline_file = "ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"


def download_config_file() -> None:
    # Download training configuration file for model
    download_config = f"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/{base_pipeline_file}"
    subprocess.run(["/usr/bin/wget", download_config], check=True)  # noqa: S603


# perform training
num_steps = 40000
batch_size = 16

model_name = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
pipeline_fname = f"/models/mymodel/{base_pipeline_file}"
fine_tune_checkpoint = f"/models/mymodel/{model_name}/checkpoint/ckpt-0"


def get_num_classes(pbtxt_fname) -> int:
    from object_detection.utils import label_map_util

    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=90,
        use_display_name=True,
    )
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


num_classes = get_num_classes(label_map_pbtxt_fname)
print("Total classes:", num_classes)

print("writing custom config file")

with Path(pipeline_fname).open() as f:
    s = f.read()
with Path("pipeline_file.config", "w").open() as f:
    # Set fine_tune_checkpoint path
    s = re.sub(
        'fine_tune_checkpoint: ".*?"',
        f'fine_tune_checkpoint: "{fine_tune_checkpoint}"',
        s,
    )

    # Set tfrecord files for train and test datasets
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
        f'input_path: "{train_record_fname}"',
        s,
    )
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
        f'input_path: "{val_record_fname}"',
        s,
    )

    # Set label_map_path
    s = re.sub(
        'label_map_path: ".*?"',
        f'label_map_path: "{label_map_pbtxt_fname}"',
        s,
    )

    # Set batch_size
    s = re.sub("batch_size: [0-9]+", f"batch_size: {batch_size}", s)

    # Set training steps, num_steps
    s = re.sub("num_steps: [0-9]+", f"num_steps: {num_steps}", s)

    # Set number of classes num_classes
    s = re.sub("num_classes: [0-9]+", f"num_classes: {num_classes}", s)

    # Change fine-tune checkpoint type from "classification" to "detection"
    s = re.sub(
        'fine_tune_checkpoint_type: "classification"',
        'fine_tune_checkpoint_type: "{}"'.format("detection"),
        s,
    )

    # If using ssd-mobilenet-v2, reduce learning rate (because it's too high in the default config file)
    s = re.sub("learning_rate_base: .8", "learning_rate_base: .08", s)
    s = re.sub("warmup_learning_rate: 0.13333", "warmup_learning_rate: .026666", s)

    f.write(s)


def main() -> None:
    return


if __name__ == "__main__":
    main()

__all__ = []
