import re
import os
import numpy as np
import tensorflow as tf
import random
import scipy

from keras import backend as K
from scipy import ndimage
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping#, ReduceLROnPlateau

directory = './Spatial_image_data'
scaleFactors = [2, 3, 4]
patch_size = 41
patch_num = 64
BATCH_SIZE = 64
EPOCHS = 150
networkDepth = 38

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

def slicing_images(target_image):
    OG_image = target_image
    scaleFactor = np.random.choice(scaleFactors, 1, replace=True)   ## random choice among ScaleFactors
    hor = np.size(OG_image,0)    
    ver = np.size(OG_image,1)
    OG_upsampledImage = scipy.misc.imresize((scipy.misc.imresize(OG_image, ((1/scaleFactor)*hor, (1/scaleFactor)*ver), 'bicubic')), (hor, ver),'bicubic')
    # downscale & upscale
    
    OG_image = OG_image.astype('float32') / 255.0 
    OG_upsampledImage = OG_upsampledImage.astype('float32') / 255.0 
    
    for i in range(0, patch_num):
        randx = random.randint(0, hor-41)
        randy = random.randint(0, ver-41)   ## choose random num
        
        sliced_image = OG_image[randx:randx+41, randy:randy+41] ## get OG image patch from random position 
        upsampledImage = OG_upsampledImage[randx:randx+41, randy:randy+41]  ## get upsampledImage patch from the same position
        residualImage = sliced_image - upsampledImage ## get residual patch from the same position

        UpsampledImages.append(upsampledImage)
        ResidualImages.append(residualImage) ## save into a list 
        
def load_images(directory):
    for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    filepath = os.path.join(root, filename) ## get the file path
                    image = ndimage.imread(filepath, mode="YCbCr")  ## img read, YCbCr 
                    Luminance = image[:, :, 0]   ## get Y only
                    slicing_images(Luminance)   ## get patches
        
UpsampledImages = []
ResidualImages = []
    
load_images(directory)

UpsampledImages = np.array(UpsampledImages)
array_shape = np.append(UpsampledImages.shape[0:3], 1)
UpsampledImages = np.reshape(UpsampledImages, (array_shape))    

ResidualImages = np.array(ResidualImages)
array_shape = np.append(ResidualImages.shape[0:3], 1)
ResidualImages = np.reshape(ResidualImages, (array_shape)) 

n = len(UpsampledImages)
Train_num = round(n*0.7)

Train_UpsampledImages = UpsampledImages[0:Train_num, :, :, :] 
Train_ResidualImages = ResidualImages[0:Train_num, :, :, :]

Valid_UpsampledImages = UpsampledImages[Train_num+1:n-1, :, :, :]
Valid_ResidualImages = ResidualImages[Train_num+1:n-1, :, :, :]


#################################################################################################
     
input_shape = (41, 41, 1) # patch size = 41X41

input_img = Input(shape=input_shape)

x = BatchNormalization()(input_img)

x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
x = Activation('relu')(x)

for i in range(1, networkDepth-1, 2):
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Activation('relu')(x)
    
x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(x)    

model = Model(inputs=input_img, outputs=x, name='VDSR')

#model.load_weights('./checkpoints/weights-improvement-20-26.93.hdf5')

adam = Adam(lr=0.001)
model.compile(adam, loss='mse', metrics=["accuracy", PSNR])

model.summary()

checkpointer = ModelCheckpoint(filepath='vdsr_my_weight.h5', monitor=PSNR, verbose=1, mode='max', save_best_only=True)
earlyStopper = EarlyStopping(monitor='val_PSNR', min_delta=0, patience=5, verbose=1, mode='auto')

model.fit(x=Train_UpsampledImages, y=Train_ResidualImages, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(Valid_UpsampledImages, Valid_ResidualImages), shuffle=True, callbacks = [checkpointer, earlyStopper])

print("Done training!!!")

print("Saving the final model ...")

model.save('vdsr_my_weight.h5')  # creates a HDF5 file 
del model  # deletes the existing model

