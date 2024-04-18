import argparse  # noqa: INP001, D100
from pathlib import Path

import tensorflow as tf

path = "pic"


def gen_data():
    dataset_list = tf.data.Dataset.list_files(path + "/*.jpg")
    for _ in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [300, 300])
        image = tf.cast(image / 255.0, tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=Path, help="input model dir")
    parser.add_argument(
        "--output_path",
        "-o",
        type=Path,
        help="output tflite model path",
    )

    args = parser.parse_args()
    input_path: Path = args.input_path
    output_path: Path = args.output_path

    converter = tf.lite.TFLiteConverter.from_saved_model(str(input_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.representative_dataset = gen_data
    tflite_model = converter.convert()

    with output_path.open("wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()


__all__ = []
