# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:32:39 2020

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

input_size=YOLO_INPUT_SIZE

while True:
    ID = random.randint(0, 200-1)
    label_txt = "mnist/mnist_test.txt"
    image_info = open(label_txt).readlines()[ID].split()

    image_path = image_info[0]
    print(image_path)
    
    # change the path to the noise folder
    #Just put r before your path copied from windows OS, it converts to recoganizd path
    noise_folder = r"C:\Users\lwang\Documents\DL\TensorFlow-2.x-YOLOv3-master\mnist\mnist_test_noise"
    ID = image_path.split("\\")[-1] # image ID
    image_path =os.path.join(noise_folder, ID)
    print(image_path)
    

    yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
    yolo.load_weights("./checkpoints/yolov3_custom_Tiny") # use keras weights

    detect_image(yolo, image_path, "mnist_test.jpg", input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    time.sleep(1)

