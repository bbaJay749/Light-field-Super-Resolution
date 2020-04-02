from keras.models import Model
from keras.layers import Activation
from keras.layers import Conv2D, Input, BatchNormalization
from scipy.misc import imsave
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from PIL import Image

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

V = 960    # wanted Resolution
H = 1280
input_shape = (V, H, 1)

networkDepth = 38
input_img = Input(shape = input_shape)

x = BatchNormalization()(input_img)

x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
x = Activation('relu')(x)

for i in range(1, networkDepth-1, 2):
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Activation('relu')(x)
    
x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(x)    

model = Model(inputs=input_img, outputs=x, name='VDSR')

model.load_weights('D:/Dropbox (3dlab Inha)/vdsr_my_weight.h5')


## spatially SR all the N X N light field data. numbering considering Angular SR X 2 

index_counter = 1
angular_resolution = 9
OG_counter = 1

for i in range(1, pow(angular_resolution, 2) + 1):
            global index_counter
            global OG_counter
            
            Iycbcr= Image.open('C:/Users/my/Desktop/LF_SR_TEMP/buddha/' + str(OG_counter) + '.png').convert('YCbCr')
            OG_counter += 1
            
            Iy_bicubic, Icb_bicubic, Icr_bicubic = Iycbcr.split()       # split into I, Cb, Cr
           
#            Iy_bicubic = Iy_bicubic.resize((round(H/2), round(V/2)), Image.BICUBIC)
#            Icb_bicubic = Icb.resize((round(H/2), round(V/2)), Image.BICUBIC)
#            Icr_bicubic = Icr.resize((round(H/2), round(V/2)), Image.BICUBIC)
            
            Iy_bicubic = Iy_bicubic.resize((H, V), Image.BICUBIC)   # resize into desried resolution 
            Icb_bicubic = Icb_bicubic.resize((H, V), Image.BICUBIC)
            Icr_bicubic = Icr_bicubic.resize((H, V), Image.BICUBIC)
            
            Iy_bicubic = np.array(Iy_bicubic)   # nparray
            Icb_bicubic = np.array(Icb_bicubic)
            Icr_bicubic = np.array(Icr_bicubic)
            
            Iy_bicubic = np.expand_dims(Iy_bicubic, axis=0)     # to be fitted to input size
            Iy_bicubic = np.expand_dims(Iy_bicubic, axis=3)
            Iy_bicubic = im2double(Iy_bicubic)
            
            Iresidual = model.predict(Iy_bicubic)   # predict
            
            Iy_bicubic = np.reshape(Iy_bicubic, (V, H)) 
            Iresidual = np.reshape(Iresidual, (V, H))
            
            Isr = Iy_bicubic + Iresidual    # core idea : original Y + predicted Y differance
            
            Ivdsr = np.zeros((V, H, 3))
            
            Ivdsr[:, :, 0] = Isr * 255
            Ivdsr[:, :, 1] = Icb_bicubic 
            Ivdsr[:, :, 2] = Icr_bicubic    # make Y, Cb, Cr into single one
            
            Ivdsr = ycbcr2rgb(Ivdsr)
            
            Ivdsr = Image.fromarray(Ivdsr.astype('uint8'))
            
            plt.imshow(Ivdsr)
            
            imsave('C:/Users/my/Desktop/OSK TEMP/truck_spatial_SR_9X9/truck_ (' + str(index_counter) + ').png' , Ivdsr)
            
            if(index_counter % ((2*angular_resolution)-1) == 0):
                index_counter = index_counter + (2*angular_resolution)
            else:
                index_counter = index_counter + 2 


## simply resize images using bicubic interpolation method 
                
index_counter = 0
angular_resolution = 9
OG_counter = 1

for i in range(1, pow(angular_resolution, 2) + 1):
            global index_counter
            index_counter += 1
            Iycbcr= Image.open('C:/Users/my/Desktop/OSK TEMP/truck_original_9X9/truck (' + str(index_counter) + ').png')
            
            Iycbcr = Iycbcr.resize((640, 480), Image.BICUBIC)
            
            imsave('C:/Users/my/Desktop/OSK TEMP/truck_bicubic_x0.5_9X9/truck_' + str(index_counter) + '.png' , Iycbcr)
