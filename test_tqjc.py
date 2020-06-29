import os
from tensorflow.keras import layers
from tensorflow.keras import Model
# from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from data import ShanghaitechDataset
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation, Dense, BatchNormalization
from tensorflow.keras.models import Model
from model_rqjc import MSCNN
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import time


last_output=1
pre_trained_model=1
model=1
x=1
base_dir=1
train_dir=1
validation_dir=1

train_datagen=1
test_datagen=1
train_generator=1
validation_generator=1


def pre_M_model():
    global pre_trained_model
    global last_output
    global model
    global x

    model = MSCNN((112, 112, 3))
    local_weights_files = 'E:/best_model_weights.h5'
    model.load_weights(local_weights_files)
    for layer in model.layers:
        layer.trainable=False
    model.compile(optimizer=SGD(lr=3e-4, momentum=0.9), loss='mse')



def trainin():
    history = model.fit(
        ShanghaitechDataset().gen_train(10, 112),
        steps_per_epoch=ShanghaitechDataset().get_train_num()//10,
        # batchsize=2,
        epochs=int(0))

if __name__ == '__main__':

    pre_M_model()
    trainin()
    export_dir = 'E:/pythonn/learn'
    tf.saved_model.save(model, export_dir)
    filename='E:/dataest/pe/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images/IMG_2.jpg'
    imgp=cv2.imread(filename)
    # time_start = time.time()
    img = cv2.resize(imgp, (112, 112))/255
    # cv2.imshow("img",img);
    # cv2.waitKey(0);
    # ddd=cv2.GaussianBlur(img, (15, 15), 0)
    img = np.expand_dims(img, axis=0)
    # cct=model.predict(img)
    # ccc=np.array(cct)
    # cccc=ccc.reshape((28,28,1))
    # enlarge = cv2.resize(cccc, (280, 280))
    # cv2.imshow("dd", enlarge)
    # cv2.waitKey(0)
    dmap = np.squeeze(model.predict(img), axis=-1)
    #
    dmap = cv2.GaussianBlur(dmap, (15, 15), 0)
    #
    print("dmap",np.sum(dmap))
    # time_end = time.time()
    # print('time cost', time_end - time_start, 's')