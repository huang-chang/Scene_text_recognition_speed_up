#-*- coding:utf-8 -*-
from imp import reload

from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model

import densenet

reload(densenet)

input = Input(shape=(32, None, 1), name='the_input')
y_pred = densenet.dense_cnn(input, 5990)
basemodel = Model(inputs=input, outputs=y_pred)

basemodel.load_weights('/home/vcaadmin/zhuangwu/huang/scene_text_recognition/densenet/weights_densenet-04-0.66.h5')

plot_model(basemodel, to_file='model.png')
print('OK')
