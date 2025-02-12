from flask import Flask, render_template, request, redirect
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask import flash
from flask import session
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
import os
import glob
import shutil
from PIL import Image
from keras.applications.vgg19 import VGG19

import numpy as np
import sklearn
#import Keras packages
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img
import random
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import imageio
import numpy as np
from keras import applications
from keras.layers import Input
from keras.models import Model,load_model
from keras import optimizers
from keras.utils import get_file
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import random
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

img_w = 256
img_h = 256

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


def prep_data1(mode,autoencoder):
    assert mode in {'test', 'train', 'val'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    file1 = os.listdir('static/img')
    #files = file1.read().split('.JPG')
    n = 1 
    
        
    idx = 0
    tep = random.randint(100,500)
    for filename in file1:
        print(filename)
        if(filename == ""):
            break
        new_im = Image.open('static/img/' + filename)
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
                    labeled1[i,j] = [0,0,0]
                elif(labeled[i,j] == 1):
                    labeled1[i,j] = [0, 255,0]
                else:
                    labeled1[i,j] = [128,128,0]
        
        
        imageio.imwrite('static/img1/predict_segnet'+str(tep)+'.jpg', labeled1.astype('uint8'))
        
        #new_im1 = Image.open(path + mode + '-colormap/' + filename + '.JPG')


        #img, gt = [imread(path + mode + '/' + filename + '.JPG')], imread(path + mode + '-colormap/' + filename + '.JPG')
        
        #img, gt = [np.array(new_im,dtype=np.uint8)], np.array(new_im1,dtype=np.uint8)
        #data.append(np.reshape(img,(256,256,3)))
        #label.append(label_map1(gt))
        #sys.stdout.write('\r')
        #sys.stdout.flush()
    #sys.stdout.write('\r')
    #sys.stdout.flush()
    #data, label = np.array(data), np.array(label).reshape((n, img_h, img_w, n_labels))

    #print( mode + ': OK')
    #print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    #print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    #print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return tep #data, label

#autoencoder.load_weights('model_5l_weight_ep50.hdf5')

src_dir = "static/img"
dst_dir = "static/img1"

app = Flask(__name__)
app.secret_key = "super secret key"

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route('/', methods=['GET', 'POST'])
def upload():
    flash('')
    if request.method == 'POST' and 'photo' in request.files:
        	filename = photos.save(request.files['photo'])
        	for jpgfile in glob.iglob(os.path.join(src_dir, "*.*")):
        	  shutil.copy(jpgfile, dst_dir)
        
        	autoencoder = SegNet()

        	optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
        	autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        	print( 'Compiled: OK')
        	autoencoder.summary()

        	autoencoder.load_weights('model_5l_weight_leaf_segnet.weights.h5')
        	# don't train existing weights
        	
        
        	
        	tep=prep_data1('train', autoencoder)
        
        	#arr = np.argmax(arr, axis=1)

        	"""
        	i = 0
        	j = 0
        	while(i < len(arr)):
        	  if(arr[i] == 1):
        	    j += 1
        	  i += 1
        	"""
        #flash('Images with no alcohol content found: ' + j)
        
        #for layer in classifier.layers:
        #    g=layer.get_config()
        #    h=layer.get_weights()
        #    print (g)
        #    print (h)
        
        #scores = classifier.evaluate_generator(test_set,62/32)
        	#flash(str(arr[0]))
        	#if(arr[0] == 0):
        	  #flash('Image has COVID disease')
        	#else:
        	  #flash('Image does not have COVID disease')
        
        	#K.clear_session()
        
        	K.clear_session()
            
        	context = {
                    'file1': 'static/img1/' + filename,
                    'file2': 'static/img1/predict_segnet'+str(tep)+'.jpg'
                }

        	#render_template("index.html", **context)
        
        	os.remove('static/img/' + filename)
        	return render_template('image.html', **context)
    return render_template('image.html')


if __name__ == '__main__':
    app.run(debug=True)