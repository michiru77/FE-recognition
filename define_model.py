#_ import
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad

#_ define CNN_model
class CNN_model:
    def __init__(self, arg_label_num, arg_size):
        self.class_num = arg_label_num
        self.size = arg_size
        
    def nisime_kai_gap_model(self):
        model = Sequential()
        # 0~6
        model.add(Conv2D(64, (3, 3), padding='valid', data_format="channels_last", input_shape=(self.size, self.size, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        # 7~15
        model.add(Conv2D(128, (3, 3), padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        # 16~22
        model.add(Conv2D(256, (3, 3), padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        # 23
        model.add(GlobalAveragePooling2D())
        model.add(Dense(self.class_num))
        model.add(Activation('softmax'))
        # compile
        model.compile(loss="categorical_crossentropy", optimizer=Adagrad(), metrics=["accuracy"])
        return model