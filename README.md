# C.A.T.S - Camera Assisted Tracking System

## Overview

This is a continuation of the a previous senior design project, which you can
find [here](https://github.com/A-very-Cunning-ham/CATS). Credit for the original
idea, fantastic name, and any commits not made by David Krauthamer, Brandon
Boutin, and Robert Plastina, go to the original authors.

Our goal as a team was to improve on the previously created model by optimizing
both it's speed and accuracy. We wanted to use Google Coral hardware to
accelerate the model to obtain the speed increase, hence why we switched to
using TensorFlow (Lite). To increase the accuracy we trained the model to
identify not just cats, but other animals which could be easily mistaken by a
computer for a cat. The expectation was that this would reduce the false
positive rate, and increase the accuracy of the model overall.

## Dataset Sources

- Cat and Dog breeds:
  <https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset>
- Raccoons: <https://universe.roboflow.com/raccoondetector/raccoon-finder>
- Squirrels: <https://universe.roboflow.com/project1373/squirrels-faces>
- Foxes: <https://universe.roboflow.com/fox-ewljn/fox-oljyb>
- Coyotes: <https://universe.roboflow.com/david-casas/coyote-uuhaf>

Each of the non-cat and dog datasets is available in tfrecord format, and that's
what we used for training. The cats and dogs tfrecord needed to be generated, we
the modified instructions from
[here](https://coral.ai/docs/edgetpu/retrain-detection/#configure-your-own-training-data)
in order to generate it.

From what we've been able to find online, merging multiple tfrecords together
into a single large tfrecord isn't the best practice, however it's what we chose
to do for simplicity. The script used to combine the tfrecords is
[combine_tfrecord.py](./scripts/combine_tfrecord.py). The reason for combining
the tfrecords into one is it allowed us to use the a script provided from Google
themselves
[here](https://github.com/tensorflow/models/blob/7c3724fa91776595b7e6634282c4379176166369/research/object_detection/model_main_tf2.py).
