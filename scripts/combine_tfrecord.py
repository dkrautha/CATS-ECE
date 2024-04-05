from argparse import ArgumentParser  # noqa: INP001, D100
from pathlib import Path

import tensorflow as tf


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    input_dir: Path = args.input_dir

    list_of_tfrecord_files = sorted(input_dir.glob("*.tfrecord"))
    dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)

    writer = tf.data.experimental.TFRecordWriter(args.output_path)

    writer.write(dataset)


if __name__ == "__main__":
    main()

__all__ = []
