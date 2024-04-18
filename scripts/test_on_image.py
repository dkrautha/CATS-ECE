from __future__ import annotations  # noqa: INP001, D100

import argparse
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tflite_runtime.interpreter import Interpreter
import tflite_runtime.interpreter


def load_labels(file_path: Path) -> dict[int, str]:
    def get_right_of_colon(line: str) -> str:
        return line.split(":", maxsplit=1)[1].strip()

    label_map = {}
    with file_path.open("r") as f:
        iterator = iter(f)

        for line in f:
            stripped = line.strip()
            if stripped.startswith("item {"):
                id_line = next(iterator).strip()
                name_line = next(iterator).strip()

                id_ = int(get_right_of_colon(id_line)) - 1
                name = get_right_of_colon(name_line.replace("'", ""))
                label_map[id_] = name

                _ = next(iterator)  # dropping the } at the end

    return label_map


@dataclass
class DetectedObject:
    class_id: int
    class_name: str
    score: float
    bounding_box: BoundingBox


@dataclass
class BoundingBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


class InterpreterWrapper:
    _interpreter: Interpreter
    _expected_height: int
    _expected_width: int
    _score_threshold: float
    _label_map: dict[int, str]
    _resized_image: Image.Image | None = None

    def __init__(
        self: InterpreterWrapper,
        model_path: Path,
        label_map: dict[int, str],
        score_threshold: float,
        edgetpu_lib: str | None = None,
    ) -> None:
        self._interpreter = Interpreter(
            model_path=str(model_path),
            experimental_delegates=[
                tflite_runtime.interpreter.load_delegate(edgetpu_lib, {}),
            ]
            if edgetpu_lib
            else [],
        )
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        self._expected_height, self._expected_width = input_details[0]["shape"][1:3]

        self._score_threshold = score_threshold

        self._label_map = label_map

    def _set_input(self: InterpreterWrapper, jpg_image: Image.Image) -> None:
        self._resized_image = jpg_image.resize(
            (self._expected_width, self._expected_height),
        )

        as_numpy = np.array(self._resized_image).astype(np.float32) / 255.0
        as_numpy = as_numpy[np.newaxis, :, :, :]

        input_details = self._interpreter.get_input_details()
        self._interpreter.set_tensor(input_details[0]["index"], as_numpy)

    def _get_output_tensor(self: InterpreterWrapper, index: int) -> np.ndarray:
        tensor = self._interpreter.get_tensor(
            self._interpreter.get_output_details()[index]["index"],
        )
        return np.squeeze(tensor)

    @property
    def resized_image_copy(self: InterpreterWrapper) -> Image.Image:
        if self._resized_image is None:
            msg = "Interpreter has not been invoked on an image yet."
            raise ValueError(msg)
        return self._resized_image.copy()

    def invoke(
        self: InterpreterWrapper,
        jpg_image: Image.Image,
    ) -> list[DetectedObject]:
        self._set_input(jpg_image)
        self._interpreter.invoke()

        scores = self._get_output_tensor(0)
        bounding_boxes = self._get_output_tensor(1)
        count = int(self._get_output_tensor(2))
        class_ids = self._get_output_tensor(3)

        objects = []
        for i in range(count):
            # if scores[i] < self._score_threshold:
            #     continue

            bb = bounding_boxes[i]
            ymin = int(bb[0] * self._expected_height)
            ymax = int(bb[2] * self._expected_height)
            xmin = int(bb[1] * self._expected_width)
            xmax = int(bb[3] * self._expected_width)

            objects.append(
                DetectedObject(
                    class_id=class_ids[i],
                    class_name=self._label_map[class_ids[i]],
                    score=scores[i],
                    bounding_box=BoundingBox(
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                    ),
                ),
            )

        return objects


def draw_bounding_boxes(
    img: Image.Image,
    objects: list[DetectedObject],
) -> None:
    draw = ImageDraw.Draw(img)
    for obj in objects:
        bbox = obj.bounding_box
        draw.rectangle(
            (
                (bbox.xmin, bbox.ymin),
                (bbox.xmax, bbox.ymax),
            ),
            outline="red",
        )
        draw.text(
            (bbox.xmin + 10, bbox.ymin + 10),
            f"{obj.class_name}\n{obj.score:.2f}",
            fill="red",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=Path, help="input jpg path")
    parser.add_argument("--model_path", "-m", type=Path, help=".tflite model path")
    parser.add_argument(
        "--labels_path",
        "-l",
        type=Path,
        help="labels path in pbtxt format",
    )
    parser.add_argument("--output_path", "-o", type=Path, help="output jpg path")
    parser.add_argument("--edgetpu", "-e", action="store_true")
    args = parser.parse_args()

    labels = load_labels(args.labels_path)

    interpreter = InterpreterWrapper(
        model_path=args.model_path,
        label_map=labels,
        score_threshold=0.8,
        edgetpu_lib={
            "Linux": "libedgetpu.so.1",
            "Darwin": "libedgetpu.1.dylib",
            "Windows": "edgetpu.dll",
        }[platform.system()]
        if args.edgetpu
        else None,
    )

    img_path = Path(args.input_path)
    img = Image.open(img_path)

    detected_objects = interpreter.invoke(img)
    for obj in detected_objects:
        print(obj)

    cropped_img = interpreter.resized_image_copy

    draw_bounding_boxes(cropped_img, detected_objects)

    cropped_img.save(args.output_path)


__all__ = []

if __name__ == "__main__":
    main()
