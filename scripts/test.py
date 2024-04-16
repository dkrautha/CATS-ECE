import numpy as np
from PIL import Image

from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_path = "croc.jpg"
img = Image.open(img_path).resize(
    (input_details[0]["shape"][1], input_details[0]["shape"][2]),
)
img = np.array(img).astype(np.float32) / 255.0
img = img[np.newaxis, :, :, :]

interpreter.set_tensor(input_details[0]["index"], img)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]["index"])

print(output_data)
