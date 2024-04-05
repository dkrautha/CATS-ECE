import re  # noqa: INP001, D100
from pathlib import Path


def main() -> None:
    with Path("/models/research/object_detection/packages/tf2/setup.py").open() as f:
        s = f.read()

    with Path("/models/research/setup.py").open() as f:
        s = re.sub("tf-models-official>=2.5.1", "tf-models-official==2.8.0", s)
        f.write(s)


if __name__ == "__main__":
    main()

__all__ = []
