import time
import cv2
import tensorflow as tf


def main():
    cap = cv2.VideoCapture("/dev/video0")
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        counter += 1
        if counter > 150:
            cv2.imwrite("./frame.png", frame)
            break


if __name__ == "__main__":
    main()
