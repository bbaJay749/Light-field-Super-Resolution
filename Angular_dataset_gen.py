import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import scipy
from PIL import Image
from tensorflow.keras import backend as K
from scipy import ndimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, concatenate
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import h5py
from datetime import datetime

path_Train_1 = './Angular_data_Hor/Train_L'
path_Train_2 = './Angular_data_Hor/Train_R'
path_Train_Ground = './Angular_data_Hor/Train_G'

path_Train_1_Patches = './Angular_data_Hor/Train_L - Y'
path_Train_2_Patches = './Angular_data_Hor/Train_R - Y'
path_Train_Ground_Patches = './Angular_data_Hor/Train_G - Y'

path_Val_1 = './Angular_data_Hor/Val_L'
path_Val_2 = './Angular_data_Hor/Val_R'
path_Val_Ground = './Angular_data_Hor/Val_G'

path_Val_1_Patches = './Angular_data_Hor/mixed_disparity_41/Val_L - Y'
path_Val_2_Patches = './Angular_data_Hor/mixed_disparity_41/Val_R - Y'
path_Val_Ground_Patches = './Angular_data_Hor/mixed_disparity_41/Val_G - Y'

##################### GENERATING PATCHES #####################

counter = 0;

NumofData = 1275
patch_num = 786
patch_size = 32
                
def Gen_random(directory, i):
    for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if re.search("\.(JPG|jpeg|png|bmp|tiff)$", filename):
                    filepath = os.path.join(root, filename) ## 이미지 주소 따기
                    image = Image.open(filepath, mode='r')
                    hor, ver = image.size 
                    
                    random.seed(datetime.now())
                    for k in range(0, patch_num):
                        randomseed_randx[i, k] = random.randint(0, hor-patch_size)
                        randomseed_randy[i, k] = random.randint(0, ver-patch_size)  ## 랜덤한 숫자 선정
                    i += 1

def slicing_images(target_image, savedir, i):
    global counter
    
    for k in range(0, patch_num):
        sliced_image = target_image.crop((int(randomseed_randx[i, k]), int(randomseed_randy[i, k]), int(randomseed_randx[i, k])+patch_size, int(randomseed_randy[i, k])+patch_size))
        sliced_image.save(savedir + '/patch' + str(counter) + '.png')
        counter += 1
                    
def save_images(directory, savedir, i):
    for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if re.search("\.(JPG|jpeg|png|bmp|tiff)$", filename):
                    global counter
                    filepath = os.path.join(root, filename) ## 이미지 주소 따기
                    image = Image.open(filepath, mode='r')
                    image = image.convert('YCbCr')
                    Iy, Icb, Icr = image.split()
                    del Icb
                    del Icr
                    sliced_images = slicing_images(Iy, savedir, i)   ## 자르기
                    i += 1                   
                    
randomseed_randx = np.zeros((NumofData, patch_num))
randomseed_randy = np.zeros((NumofData, patch_num))

Gen_random(path_Train_1, i = 0)

save_images(path_Train_1, path_Train_1_Patches, i = 0)
counter = 0
save_images(path_Train_2, path_Train_2_Patches, i = 0)
counter = 0
save_images(path_Train_Ground, path_Train_Ground_Patches, i = 0)

counter = 0
save_images(path_Val_1, path_Val_1_Patches, i = 0)
counter = 0
save_images(path_Val_2, path_Val_2_Patches, i = 0)
counter = 0
save_images(path_Val_Ground, path_Val_Ground_Patches, i = 0)
counter = 0
