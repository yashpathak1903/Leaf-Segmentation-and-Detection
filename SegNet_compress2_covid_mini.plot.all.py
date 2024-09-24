import numpy as np
import pandas as pd

import json
import sys
from PIL import Image, ImageOps

#from skimage.io import imread
#from matplotlib import pyplot as plt
import random

import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] ='mode=FAST_RUN,device=cpu'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

#from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.layers import Input, ZeroPadding2D
#from tensorflow.keras.layers import Activation, Flatten, Reshape
#from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate  as merge
#from tensorflow.keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
import imageio
#from tensorflow.keras import utils as  np_utils
#from keras.applications import imagenet_utils

path = 'Leaf/'
img_w = 256
img_h = 256
n_labels = 3
Tree = [128,128,0]
leaf = [0,255,0]
Unlabelled = [0,0,0]


n_train = 65
n_test = 10
n_val = 30

def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

def label_map1(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            #print(labels[r][c])
            '''
            if(labels[r][c][0] == Tree[0] and labels[r][c][1] == Tree[1] and labels[r][c][2] == Tree[2]):
                label_map[r, c, 0] = 1
            elif(labels[r][c][0] == leaf[0] and labels[r][c][1] == leaf[1] and labels[r][c][2] == leaf[2]):
                label_map[r, c, 1] = 1
            elif(labels[r][c][0] == Unlabelled[0] and labels[r][c][1] == Unlabelled[1] and labels[r][c][2] == Unlabelled[2]):
                label_map[r, c, 2] = 1
            '''
            if(labels[r][c][0] == Unlabelled[0] and labels[r][c][1] == Unlabelled[1] and labels[r][c][2] == Unlabelled[2]):
                label_map[r, c, 0] = 1
            elif(labels[r][c][0] < 25 and labels[r][c][2] < 25 and labels[r][c][1] > 220):
                label_map[r, c, 1] = 1
            elif (labels[r][c][2] < 40 and labels[r][c][0] > 100 and labels[r][c][0] < 160 and labels[r][c][1] > 100 and labels[r][c][1] < 160):
                label_map[r, c, 2] = 1
            
    return label_map


def prep_data1(mode,autoencoder):
    assert mode in {'test', 'train', 'val'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    file1 = open(path + mode + '.txt', 'r')
    files = file1.read().split('.JPG')
    n = n_train if mode == 'train' else n_test
    
    if(mode == 'val'):
        n = n_val
        
    idx = 0
    for filename in files:
        print(filename)
        if(filename == ""):
            break
        idx += 1
        if(idx > 20):
            break
        new_im = Image.open(path + mode + '/' + filename + '.JPG')
        temp = [] 
        
        temp.append(np.reshape(new_im,(img_w, img_h,3)))
        output = autoencoder.predict(np.array(temp), verbose=1)
        output = output.reshape((output.shape[0], img_w, img_h, 3))
        #stop = time.time()
        
    

        #print(stop-start)
        
        

        #imageio.imwrite('predict_unet_b_'+ imgs[0] + '.jpg', labeled1.astype('uint8'))
        #new_im = Image.new("RGB", (544, 512))
        #new_im.paste(img1, ((544-new_size[0])//2,
                            #(512-new_size[1])//2))


        #print(output)
        labeled = np.argmax(output[0], axis=-1)
        #print(labeled)
        #print(labeled)
        labeled1 = np.zeros([img_w, img_h, 3]) 
        for i in range(0,img_w):
            for j in range(0, img_h):
                if(labeled[i,j] == 0):
                    labeled1[i,j] = Unlabelled
                elif(labeled[i,j] == 1):
                    labeled1[i,j] = leaf
                else:
                    labeled1[i,j] = Tree
        imageio.imwrite('predict_segnet('+ str(idx) + ')' + '.jpg', labeled1.astype('uint8'))
        
        new_im1 = Image.open(path + mode + '-colormap/' + filename + '.JPG')


        #img, gt = [imread(path + mode + '/' + filename + '.JPG')], imread(path + mode + '-colormap/' + filename + '.JPG')
        
        img, gt = [np.array(new_im,dtype=np.uint8)], np.array(new_im1,dtype=np.uint8)
        data.append(np.reshape(img,(256,256,3)))
        label.append(label_map1(gt))
        sys.stdout.write('\r')
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h, img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label


def prep_data(mode):
    assert mode in {'test', 'train'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
    n = n_train if mode == 'train' else n_test
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(np.reshape(img,(256,256,1)))
        label.append(label_map(gt))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048544, label.nbytes / 1048544))

    return data, label

"""
def plot_results(output):
    gt = []
    df = pd.read_csv(path + 'test.csv')
    for i, item in df.iterrows():
        gt.append(np.clip(imread(path + item[1]), 0, 1))

    plt.figure(figsize=(15, 2 * n_test))
    for i, item in df.iterrows():
        plt.subplot(n_test, 4, 4 * i + 1)
        plt.title('Ground Truth')
        plt.axis('off')
        gt = imread(path + item[1])
        plt.imshow(np.clip(gt, 0, 1))

        plt.subplot(n_test, 4, 4 * i + 2)
        plt.title('Prediction')
        plt.axis('off')
        labeled = np.argmax(output[i], axis=-1)
        plt.imshow(labeled)

        plt.subplot(n_test, 4, 4 * i + 3)
        plt.title('Heat map')
        plt.axis('off')
        plt.imshow(output[i][:, :, 1])

        plt.subplot(n_test, 4, 4 * i + 4)
        plt.title('Comparison')
        plt.axis('off')
        rgb = np.empty((img_h, img_w, 3))
        rgb[:, :, 0] = labeled
        rgb[:, :, 1] = imread(path + item[0])
        rgb[:, :, 2] = gt
        plt.imshow(rgb)

    plt.savefig('result.JPG')
    plt.show()
"""

#########################################################################################################
"""
def SegNet(input_shape=(480, 384, 3), classes=13):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(64, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(128, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(256, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(512, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(512, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 3, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(classes, 1, 1, padding="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model
"""

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


def SegNet(input_shape=(256, 256, 3), classes=3):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(classes, 1, 1, padding="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model

def SegNet_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters, input_shape=(256, 256, 3), classes=3):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(layer1_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(layer2_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(layer3_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(layer4_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(layer5_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(layer6_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(layer7_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(layer8_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(classes, 1, 1, padding="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model

"""
with open('model_5l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())
"""

autoencoder = SegNet()

print('Start')
optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print( 'Compiled: OK')
autoencoder.summary()

# Train model or load weights
'''
train_data, train_label = prep_data1('train')
val_data, val_label = prep_data1('val')
nb_epoch = 100
batch_size = 2
history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(val_data, val_label))
'''
autoencoder.load_weights('model_5l_weight_leaf_segnet.weights.h5')
prep_data1('train', autoencoder)
#autoencoder.load_weights('model_5l_weight_ep50.hdf5')


