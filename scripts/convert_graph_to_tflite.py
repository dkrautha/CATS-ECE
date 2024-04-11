# Convert exported graph file into TFLite model file
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("/training/saved_model")
tflite_model = converter.convert()

with open("/training/detect.tflite", "wb") as f:
    f.write(tflite_model)
