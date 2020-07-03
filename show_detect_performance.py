# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:00:25 2020
show test performance
@author: lwang
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import detect_image
from yolov3.configs import *
import matplotlib.pyplot as plt

input_size =YOLO_INPUT_SIZE

num_row = 2
num_col = 3
num = num_row*num_col
image_pred = []
for i in range(num):  
    label_txt = "mnist/mnist_test.txt"
    image_info = open(label_txt).readlines()[i].split()
    
    image_path = image_info[0]
    print(image_path)
    
    yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
    yolo.load_weights("./checkpoints/yolov3_custom_Tiny") # use keras weights
    
    image = detect_image(yolo, image_path, "mnist_test.jpg", input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    image_pred.append(image)
    
#%%
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(image_pred[i])
plt.tight_layout()
plt.show()
