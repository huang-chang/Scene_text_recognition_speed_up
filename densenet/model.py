#-*- coding:utf-8 -*-
from __future__ import print_function
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
from lib.utils.timer import Timer
import tensorflow as tf

from keras.layers import Input
from keras.models import Model
# import keras.backend as K

from . import keys
from . import densenet

reload(densenet)

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

#input = Input(shape=(32, None, 1), name='the_input')
input = Input(shape=(32, None, 1), name='the_input')
y_pred= densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)

modelPath = os.path.join(os.getcwd(), 'densenet/models/weights_densenet.h5')
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)
    #temp_image = np.zeros((32,280))+0.1
    #_ = basemodel.predict(temp_image.reshape([1, 32, 280, 1]))
    #tf.get_default_graph().finalize()
    #print('ok')
    basemodel._make_predict_function()

def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)


def predict(img):
    
    timer = Timer()
    timer.tic()
    y_pred = basemodel.predict(img,batch_size=1)
    #y_pred = basemodel.predict(X)
    timer.toc()
    print(img.shape, timer.total_time)
    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)
    
    return out
