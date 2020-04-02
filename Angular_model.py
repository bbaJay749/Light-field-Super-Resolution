import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import scipy

from keras import backend as K
from scipy import ndimage
from keras.models import Model
from keras.layers import Input, Conv2D, Conv3D, BatchNormalization, Activation, concatenate, Dropout
from keras.optimizers import Adam, rmsprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import TruncatedNormal

patch_size = 32
BATCH_SIZE = 256
EPOCHS = 5000
networkDepth = 2
counter=0

path_Train_1_Patches = './Angular_data_Hor/Train_L - Y' 
path_Train_2_Patches = './Angular_data_Hor/Train_R - Y'
path_Train_Ground_Patches = './Angular_data_Hor/Train_G - Y'

path_Val_1_Patches = './Angular_data_Hor/mixed_disparity/patches/Val_L - Y'
path_Val_2_Patches = './Angular_data_Hor/mixed_disparity/patches/Val_R - Y'
path_Val_Ground_Patches = './Angular_data_Hor/mixed_disparity/patches/Val_G - Y'

TruncatedNormal(mean=0.0, stddev=0.01)

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

def load_images(directory, listname):
    for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    filepath = os.path.join(root, filename) ## 이미지 주소 따기
                    image = ndimage.imread(filepath)
                    listname.append(image)
                    global counter
                    if(counter%10000 == 0):
                        print(counter)
                    counter += 1
                    
Train_1_Patches = []
Train_2_Patches = []
Train_Ground_Patches = []
    
load_images(path_Train_1_Patches, Train_1_Patches)
load_images(path_Train_2_Patches, Train_2_Patches)
load_images(path_Train_Ground_Patches, Train_Ground_Patches)

size_of_data = len(Train_1_Patches)
tmp = [[x,y,z] for x, y, z in zip(Train_1_Patches, Train_2_Patches, Train_Ground_Patches)]

np.random.shuffle(tmp)
Train_1_Patches = [n[0] for n in tmp]
Train_2_Patches = [n[1] for n in tmp]
Train_Ground_Patches = [n[2] for n in tmp]
del tmp

Train_1_Patches = np.expand_dims(Train_1_Patches, axis=3)
Train_1_Patches = np.expand_dims(Train_1_Patches, axis=4)
Train_1_Patches = (np.array(Train_1_Patches))/255.0

Train_2_Patches = np.expand_dims(Train_2_Patches, axis=3)
Train_2_Patches = np.expand_dims(Train_2_Patches, axis=4)
Train_2_Patches = (np.array(Train_2_Patches))/255.0

Train_Ground_Patches = np.expand_dims(Train_Ground_Patches, axis=3)
Train_Ground_Patches = np.expand_dims(Train_Ground_Patches, axis=4)
Train_Ground_Patches = (np.array(Train_Ground_Patches))/255.0

##########################################################################################
#
#Val_1_Patches = []
#Val_2_Patches = []
#Val_Ground_Patches = []
#    
#load_images(path_Val_1_Patches, Val_1_Patches)
#load_images(path_Val_2_Patches, Val_2_Patches)
#load_images(path_Val_Ground_Patches, Val_Ground_Patches)
#
#Val_1_Patches = (np.array(Val_1_Patches))/255.0
#Val_1_Patches = np.expand_dims(Val_1_Patches, axis=3)
#
#Val_2_Patches = (np.array(Val_2_Patches))/255.0
#Val_2_Patches = np.expand_dims(Val_2_Patches, axis=3)
#
#Val_Ground_Patches = (np.array(Val_Ground_Patches))/255.0
#Val_Ground_Patches = np.expand_dims(Val_Ground_Patches, axis=3)

###################################################################

input_shape = (patch_size, patch_size, 1) #(None)

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

adam = Adam(lr=0.000001,beta_1=0.9, beta_2=0.999)

model.load_weights('Angular_model_conv2d_depth2_patchsize32_CB.h5')

model.compile(adam, loss= 'mse', metrics=[PSNR])

checkpointer = ModelCheckpoint(filepath='Angular_model_conv2d_depth2_patchsize32_CB.h5', verbose=1, save_best_only=True)
earlyStopper = EarlyStopping(monitor='PSNR', min_delta=0, patience=500, verbose=1, mode='auto')

model.summary()

model.fit(x=[Train_1_Patches, Train_2_Patches], y=Train_Ground_Patches, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True, 
          callbacks = [checkpointer, earlyStopper], validation_split = 0.1)
#          validation_data = ([Val_1_Patches, Val_2_Patches], Val_Ground_Patches), initial_epoch = 0)

print("Done training!!!")

print("Saving the final model ...")

model.save('Angular_model_mixed_disparity_41_depth15_10_3.h5')  # creates a HDF5 file 

#model.evaluate((Test_L_Patches, Test_R_Patches), Test_Ground_Patches)

