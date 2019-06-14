from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.preprocessing.image import load_img, img_to_array

import os
#import cPickle
import numpy as np
import six
import csv

import sys

#from model_cnn_nisime_kai import *
#from model_cnn_test_gap import *
#from model_cnn_nisime_kai_gap import *
#from model_cnn_exam import *
#from model_cnn_exam_add_bn import *
#from model_cnn_chav import *
#from model_cnn_chav_addbn import *
#from model_cnn_nisime_kai_gap_custum import *
from define_model import CNN_model

def gcn(img):
    img = img - np.mean(img)
    img = img / np.std(img)
    return (img)

def load_label():
    labels = []
    with open("./kaggle_nisime_gap_i150_result/binary_label.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(row[1])
    return labels

def image_load(csvname, image_size, label_num):
    data = []
    label = []
    if not os.path.isfile(csvname):
        print ("%s is not found" %(csvname))
        exit()
    with open(csvname, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            img = load_img(row[0], grayscale=True, target_size=(image_size,image_size))
            img = img_to_array(img)/255.
            #img = gcn(img)
            data.append(img)
            if row[1] == 0:
                label.append(1)
            else:
                label.append(0)
    data = np.array(data)
    #one_hot_lab= np_utils.to_categorical(label)
    one_hot_lab= np_utils.to_categorical(label, label_num)
    label = np.array(one_hot_lab)
    return data, label


if __name__ == '__main__':
    
    #im_size = 128
    im_size = 48
    #epoch = 50
    epoch = 150
    #epoch = 100
    batch = 128
    
    label = load_label()
    label_num = len(label)
    
    train_csv = './kaggle_nisime_gap_i150_result/train_image.csv'
    test_csv = './kaggle_nisime_gap_i150_result/test_image.csv'
    
    X_train, y_train = image_load(train_csv, im_size, label_num)
    X_test, y_test = image_load(test_csv, im_size, label_num)


    ins_get_model = CNN_model(label_num, im_size)
    model = ins_get_model.nisime_kai_gap_binary_model()
    #model = Alexnet(label_num, im_size)
    #model = cnn_gap_model(label_num, im_size)
    
    check = ModelCheckpoint("train_model.hdf5")
    csv_logger = CSVLogger('train_log.csv')
    
    model.summary()
    hist = model.fit(X_train, y_train, batch_size=batch, epochs=epoch, verbose=1, validation_data=(X_test, y_test), callbacks=[check, csv_logger])
