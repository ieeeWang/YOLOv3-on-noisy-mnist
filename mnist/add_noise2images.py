# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:45:13 2020
add Guassion noise to images
@author: lwang
"""
import os
import glob
import random
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage 

#%% batch process under a foler
# folder_in =  "mnist_test"
folder_in =  "mnist_train"
folder_out = f"{folder_in}_noise"

paths = glob.glob(os.path.join(folder_in, '*.jpg'))
paths.sort()
for path in paths:
    print(path)
    # read image
    image = cv2.imread(path)
    # add gaussian noise
    image_noise = skimage.util.random_noise(image, mode='gaussian', mean=.2, var=.5)
    image_noise *= 255 # range 1 -> 255
    image_noise = np.uint8(image_noise) # for yolo-tiny
    # save image to folder
    ID = path.split("\\")[-1] # "\\" for windows
    savepath = os.path.join(folder_out, ID)
    cv2.imwrite(savepath, image_noise)   
  
    
fig = plt.figure(figsize=(5, 10))
plt.imshow(image_noise)
plt.show()
      


#%% Lei: check raw images
fig = plt.figure(figsize=(5, 10))
# plt.imshow(image, cmap='gray')
plt.imshow(image)
plt.show()


im_arr = np.asarray(image)
im_arr_noise = skimage.util.random_noise(im_arr, mode='gaussian', mean=.2, var=.5)
fig = plt.figure(figsize=(5, 10))
plt.imshow(im_arr_noise)
plt.show()
