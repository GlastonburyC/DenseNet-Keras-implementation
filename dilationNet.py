from keras.layers import Conv2D, BatchNormalization, Dense, Dropout
from keras.layers import Concatenate, AveragePooling2D, PReLU, Input, Flatten
from keras import losses
from keras.models import Model

import os, sys, wget
import tarfile

from keras.datasets import cifar10
from keras.optimizers import Adam

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


# load in the CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

inputs = Input(shape=x_train.shape[1:])


x = Conv2D(32,(3,3),padding='same', dilation_rate = 1)(inputs)
x = AveragePooling2D((2,2))(x)
x = BatchNormalization()(x)
x = PReLU(alpha_initializer='zeros')(x)

x = Dropout(p=0.5)(x)

x = Conv2D(32,(3,3),padding='same', dilation_rate = 2)(x)
x = AveragePooling2D((2,2))(x)
x = BatchNormalization()(x)
x = PReLU(alpha_initializer='zeros')(x)

x = Dropout(p=0.5)(x)


x = Conv2D(32,(3,3),padding='same', dilation_rate = 2)(x)
x = AveragePooling2D((2,2))(x)
x = BatchNormalization()(x)
x = PReLU(alpha_initializer='zeros')(x)

x = Dropout(p=0.5)(x)

x = Conv2D(32,(3,3),padding='same', dilation_rate = 2)(x)
x = AveragePooling2D((2,2))(x)
x = BatchNormalization()(x)
x = PReLU(alpha_initializer='zeros')(x)

x = Dropout(p=0.5)(x)

x = Flatten()(x)
predictions = Dense(10, activation='softmax')(x)




model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs=200

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=128),
                        steps_per_epoch=int(np.ceil(x_train.shape[0] / float(128))),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=8)

