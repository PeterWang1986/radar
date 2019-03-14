# Single Image Crowd Counting via Multi Column Convolutional Neural Network

This repository supply a TensorFlow implementation of CVPR2016 paper [Single Image Crowd Counting via Multi Column Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf).

The code follow the design idea of this work [SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow), and some code are copied from there.

## Models
All the models are checked in logs/, and here are the performances test from my side:

| model | training data | testing data | MAE | RMSE | notes |
|-------|:-------------:|:------------:|:---:|:---:|:-----:|
| model_v6 | shtech_part_B_train | shtech_part_B_test | 20.7 | 33.9 |
