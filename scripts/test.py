import numpy as np  # noqa: INP001, D100
from PIL import Image, ImageDraw
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_path = "croc.jpg"
cropped_image = Image.open(img_path).resize(
    (input_details[0]["shape"][1], input_details[0]["shape"][2]),
)
img = np.array(cropped_image).astype(np.float32) / 255.0
img = img[np.newaxis, :, :, :]

interpreter.set_tensor(input_details[0]["index"], img)
interpreter.invoke()

# output_data = interpreter.get_tensor(output_details[0]["index"])

from pprint import pprint

pprint(output_details)

scores = np.squeeze(interpreter.get_tensor(output_details[0]["index"]))
assumed_cat_score = scores[0]

bounding_boxes = np.squeeze(interpreter.get_tensor(output_details[1]["index"]))
assumed_cat_box = bounding_boxes[0]

unknown = np.squeeze(interpreter.get_tensor(output_details[2]["index"]))

classes_ids = np.squeeze(interpreter.get_tensor(output_details[3]["index"]))
cat_id = classes_ids[0]

print(f"scores: {scores}")
print(f"cat score: {assumed_cat_score}")
print(f"boxes: {bounding_boxes}")
print(f"cat box: {assumed_cat_box}")
print(f"unknown: {unknown}")
print(f"classes_ids: {classes_ids}")
print(f"cat id: {cat_id}")

ymin, xmin, ymax, xmax = assumed_cat_box

_, height, width, _ = interpreter.get_input_details()[0]["shape"]

ymin = int(ymin * height)
ymax = int(ymax * height)
xmin = int(xmin * width)
xmax = int(xmax * width)

print(f"{ymin=}")
print(f"{ymax=}")
print(f"{xmin=}")
print(f"{xmax=}")

bb = cropped_image.convert("RGB")
draw = ImageDraw.Draw(bb)
draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red")
bb.save("output.jpg")
