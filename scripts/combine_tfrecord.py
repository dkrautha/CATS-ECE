from __future__ import annotations  # noqa: INP001, D100

from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf


def main() -> None:
    parser = ArgumentParser()
    # receives a list of input paths
    parser.add_argument("input_paths", type=Path, nargs="+")
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    input_paths: list[Path] = args.input_paths

    list_of_tfrecord_files = sorted(input_paths)
    dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)

    output_path: Path = args.output_path
    with tf.io.TFRecordWriter(str(output_path)) as f:
        for record in dataset:
            f.write(record.numpy())  # type: ignore


if __name__ == "__main__":
    main()

__all__ = []
