# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:13:51 2022

@author: joanp
"""

import time
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def get_train_test():
    input_path = r'C:/Users/joanp/OneDrive/Documentos/UAB_Enginyeria_de_dades/2n_curs_2n_semestre/Psiv/labs/lab1/lab1/highway/input/'
    train = []
    test = []
    for i in range(1050,1350,1):
        filename = input_path + '\in00' + str(i) + '.jpg'
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if i < 1200:
            train.append(image)
        else:
            test.append(image)
    return np.array(train), np.array(test)

train, test = get_train_test()
print(train)
# MEAN IMAGE
def get_mean_image():
    print("Calculating mean...")
    

    t = time.time()
    mean_image = train.mean(axis=0)
    print("Time:", time.time() - t)
    
    return mean_image

mean_image = get_mean_image()

# STD IMAGE
def get_std_image():
    print("Calculating std...")
    

    t = time.time()
    std_image = np.std(train, axis=0)
    print("Time:", time.time() - t)
    return std_image

std_image = get_std_image()

# MEAN SEGMENTATION
def segmentation1():
    print("Image - mean_image segmentation...")
    car_mean_segmenation = abs(train[0] - mean_image)
    car_mean_segmenation = (50 > car_mean_segmenation)
    plt.imshow((car_mean_segmenation * 255).astype(np.uint8), cmap="gray")


aplha = 2
beta = 10

# SEGMENTATION
def segmentation2():
    print("Image - mean_image > alpha * std_image + beta segmentation...")
    im1 = abs(train[0] - mean_image)
    plt.figure(0)
    plt.title("Image - mean_image")
    plt.imshow(im1, cmap="gray")
    plt.show()
    im2 = aplha * std_image + beta
    plt.figure(1)
    plt.title("alpha * std_image + beta")
    plt.imshow(im2, cmap="gray")
    plt.show()
    car_segmentation = im1 > im2
    plt.figure(2)
    plt.imshow((car_segmentation * 255).astype(np.uint8), cmap="gray")
    plt.show()

# VIDEO
def create_video():
    print("Creating video...")
    t = time.time()
    from skimage import morphology
    import scipy.ndimage as ndi
    video = []
    kernel = np.ones((5,5))
    for img in test:
        new_img = ((abs(img - mean_image) > 50) * 255).astype(np.uint8)
        new_img[:100,:100] = False
        new_img = morphology.remove_small_objects(new_img.astype(bool), \
                                                    min_size=20, connectivity=2).astype(int)
        new_img *= 255
        new_img = new_img.astype(np.uint8)
        
        new_img = cv2.dilate(new_img, kernel, iterations = 1)
        new_img = cv2.erode(new_img, kernel, iterations = 1)
        
        new_img = ndi.morphology.binary_fill_holes(new_img, structure=np.ones((3,4)))
        
        new_img = new_img.astype(np.uint8)
        new_img *= 255
        
        
        video.append(new_img)
        
    x, y = test[0].shape
    local_path = os.path.dirname(os.path.abspath('__file__'))
    out = cv2.VideoWriter(local_path + "/mask_video.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, (y, x), False)        
    for i in range(len(video)):
        out.write(video[i])
    out.release()
    print("Done! Time:", time.time() - t)
    
    return video
        
video = create_video()      
  
def create_real_video():
## EVALUATE 
    grd_path = r'C:/Users/joanp/OneDrive/Documentos/UAB_Enginyeria_de_dades/2n_curs_2n_semestre/Psiv/Practica/frames_video/'
    print(grd_path)
    real = []
    for i in range(0,514,1):
        filename = grd_path + 'gt00' + str(i) + '.png'
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        print(image)
        real.append(image)
    #print(real)  
    local_path = os.path.dirname(os.path.abspath('__file__'))
    x, y = test[0].shape
    out = cv2.VideoWriter(local_path + "/test_mask.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, (y, x), False)    

    for i in range(len(real)):
        out.write(real[i])
    out.release()
    
    return real

real = create_real_video()

print("Evaluating mse...")
array = []
for x, y in zip(video, real):
    mse = ((x - y)**2).mean(axis=None)
    array.append(mse)
print("MSE:", np.mean(array))