from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, Conv3D, MaxPooling2D, Input, ZeroPadding2D, BatchNormalization, add, concatenate, Dropout
from keras.preprocessing import image
from scipy.misc import imsave
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from PIL import Image
from PIL import ImageOps
import scipy.io as sio
import cv2
import skimage
from scipy import ndimage
from cv2 import imread, resize, INTER_CUBIC
import re
import os

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)  
    return out

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def img_processing(im):
    im_Iy, im_Icb, im_Icr = im.split()  # 색 영역 split
    
    im_Iy = np.array(im_Iy)     # numpy array화
    im_Icb = np.array(im_Icb)
    im_Icr = np.array(im_Icr)
    
    im_Iy = np.expand_dims(im_Iy, axis=0)   # 학습가능한 shape로
    im_Iy = np.expand_dims(im_Iy, axis=3)
#    im_Iy = np.expand_dims(im_Iy, axis=4)
    im_Iy = im_Iy/255.0
        
    im_Icb = np.expand_dims(im_Icb, axis=0)
    im_Icb = np.expand_dims(im_Icb, axis=3)
  #  im_Icb = np.expand_dims(im_Icb, axis=4)
    im_Icb = im_Icb/255.0

    im_Icr = np.expand_dims(im_Icr, axis=0)
    im_Icr = np.expand_dims(im_Icr, axis=3)
   # im_Icr = np.expand_dims(im_Icr, axis=4)
    im_Icr = im_Icr/255.0    
    
    return im_Iy, im_Icb, im_Icr

networkDepth = 2
V = 960  
H = 1280

input_shape = (V, H, 1)


input_img_1 = Input(shape=input_shape)
input_img_2 = Input(shape=input_shape)

input_img1 = BatchNormalization()(input_img_1)
input_img2 = BatchNormalization()(input_img_2) 

x = concatenate([input_img1, input_img2])
x = Conv2D(64, (9, 9), padding='same', kernel_initializer='TruncatedNormal')(x)    
x = Activation('relu')(x)

x = Conv2D(32, (5, 5), padding='same', kernel_initializer='TruncatedNormal')(x)     
x = Dropout(0.5)(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same', kernel_initializer='TruncatedNormal')(x)
x = Dropout(0.5)(x)
x = Activation('relu')(x)
    
for i in range(0, networkDepth):
    res = x
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='TruncatedNormal')(x)     
    x = Conv2D(32, (5, 5), padding='same', kernel_initializer='TruncatedNormal')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (5, 5), padding='same', kernel_initializer='TruncatedNormal')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='TruncatedNormal')(x)     
    x = concatenate([x, res])
    
x = Conv2D(32, (3, 3), padding='same', kernel_initializer='TruncatedNormal')(x)  
x = Conv2D(1, (3, 3), padding='same', kernel_initializer='TruncatedNormal')(x)  


model = Model(inputs=[input_img_1, input_img_2], outputs=x, name='Angular_ver')

model.load_weights('Angular_model_conv3d_depth1_patchsize32_small.h5')

####################################hor angular SR#######################################
index_counter = -1
Angular_Resolution = 9

for i in range(1, pow(Angular_Resolution, 2) - (Angular_Resolution - 1)):
        if((index_counter + 2) % ((2*Angular_Resolution) - 1) == 0):
            index_counter = index_counter + (2*Angular_Resolution) + 2  
        else:
            index_counter = index_counter + 2 
            
        Image_1 = Image.open('C:/Users/my/Desktop/OSK TEMP/truck_total_SR_17X17_2/truck_ (' + str(index_counter) + ').png').convert('YCbCr')
        Image_2 = Image.open('C:/Users/my/Desktop/OSK TEMP/truck_total_SR_17X17_2/truck_ (' + str(index_counter + 2) + ').png').convert('YCbCr')

        Image_1_Iy, Image_1_Icb, Image_1_Icr = img_processing(Image_1)
        Image_2_Iy, Image_2_Icb, Image_2_Icr = img_processing(Image_2)
        
        Iy_Predicted = model.predict([Image_1_Iy, Image_2_Iy])
        Icb_Predicted = model.predict([Image_1_Icb, Image_2_Icb])
        Icr_Predicted = model.predict([Image_1_Icr, Image_2_Icr])
        
        Iy_Predicted = np.reshape(Iy_Predicted, (V, H))     
        Icb_Predicted = np.reshape(Icb_Predicted, (V, H))
        Icr_Predicted = np.reshape(Icr_Predicted, (V, H))
        
        Final = np.zeros((V, H, 3))
        
        Final[:, :, 0] = Iy_Predicted * 255.0
        Final[:, :, 1] = Icb_Predicted * 255.0
        Final[:, :, 2] = Icr_Predicted * 255.0
        
        Final = ycbcr2rgb(Final)
        
        Final = Image.fromarray(Final.astype('uint8'))
        
        imsave('C:/Users/my/Desktop/OSK TEMP/truck_total_SR_17X17_2/truck_ (' + str(index_counter + 1) + ').png', Final)


####################################ver angular SR#######################################
index_counter = 0

for i in range(1, (Angular_Resolution*(Angular_Resolution - 1) + (2*Angular_Resolution))):
        if(((index_counter) % ((2*Angular_Resolution)-1) == 0) and (index_counter != 0)):
            index_counter = index_counter + (2*Angular_Resolution)  
        else:
            index_counter = index_counter + 1 
            
        Image_1 = Image.open('C:/Users/my/Desktop/OSK TEMP/truck_total_SR_17X17/truck_ (' + str(index_counter) + ').png').convert('YCbCr')
        Image_2 = Image.open('C:/Users/my/Desktop/OSK TEMP/truck_total_SR_17X17/truck_ (' + str(index_counter + 2*(2*Angular_Resolution - 1)) + ').png').convert('YCbCr')
        
        Image_1.rotate(90)
        Image_2.rotate(90)
        
        Image_1_Iy, Image_1_Icb, Image_1_Icr = img_processing(Image_1)
        Image_2_Iy, Image_2_Icb, Image_2_Icr = img_processing(Image_2)
        
        Iy_Predicted = model.predict([Image_1_Iy, Image_2_Iy])
        Icb_Predicted = model.predict([Image_1_Icb, Image_2_Icb])
        Icr_Predicted = model.predict([Image_1_Icr, Image_2_Icr])
        
        Iy_Predicted = np.reshape(Iy_Predicted, (V, H)) 
        Icb_Predicted = np.reshape(Icb_Predicted, (V, H))
        Icr_Predicted = np.reshape(Icr_Predicted, (V, H))
        
        Final = np.zeros((V, H, 3))
        
        Final[:, :, 0] = Iy_Predicted * 255.0
        Final[:, :, 1] = Icb_Predicted * 255.0
        Final[:, :, 2] = Icr_Predicted * 255.0
        
        Final = ycbcr2rgb(Final)
        
        Final = Image.fromarray(Final.astype('uint8'))
        
        imsave('C:/Users/my/Desktop/OSK TEMP/truck_total_SR_17X17/truck_ (' + str(index_counter + (2*Angular_Resolution) - 1) + ').png', Final)