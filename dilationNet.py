from keras.layers import Conv2D, BatchNormalization, Dense, Dropout, merge, ZeroPadding2D
from keras.layers import concatenate, GlobalAveragePooling2D,MaxPooling2D, PReLU, Input, Flatten,AveragePooling2D
from keras import losses
from keras.models import Model
import keras
import os, sys, wget
import tarfile

from keras.datasets import cifar10
from keras.optimizers import Adam, SGD

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

def DenseBlock(x,no_layers,stage,feature_size,k):
    nb_feat=0
    concat_layers = []
    for i in range(0,no_layers):
        nb_feat += k
        concat_layers.append(ConvBlock(inputs=x,name_lyr='concat'+str(np.random.random_integers(0,1000000,1)[0]),k=nb_feat,feature_size=feature_size))
        print('Adding denselayer {}'.format(i))
    return concat_layers


def transitionLayer(x):
    x = BatchNormalization()(x)
    x = Conv2D(32,(1,1),padding='same', dilation_rate = 1)(x)
    x = Dropout(p=0.5)(x)
    out = AveragePooling2D((2,2))(x)
    return out

def ConvBlock(inputs,name_lyr,feature_size,k):
    inputs = BatchNormalization()(inputs)
    inputs = PReLU(alpha_initializer='zeros')(inputs)
    inputs = Conv2D(32,(1,1),padding='same', dilation_rate = 1)(inputs)
    inputs = BatchNormalization()(inputs)
    inputs = PReLU(alpha_initializer='zeros')(inputs)
    inputs = Dropout(p=0.5)(inputs)
    inputs = ZeroPadding2D((1, 1))(inputs)
    out = Conv2D(k,(3,3),padding='same', dilation_rate = 1)(inputs)
    return out

k = 12

inputs = Input(shape=x_train.shape[1:])

x = ZeroPadding2D((4, 4), name='conv1_zeropadding')(inputs)
x = Conv2D(32,(6,6),padding='same', dilation_rate = 1,name='init_conv')(x)
x = BatchNormalization()(x)
x = PReLU(alpha_initializer='zeros')(x)
x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
x = MaxPooling2D((2,2))(x)

dense_lst = DenseBlock(x,no_layers=6,stage=1,feature_size=32,k=k)
x = concatenate(dense_lst)
x = transitionLayer(x)

dense_lst2 = DenseBlock(x,no_layers=12,stage=2,feature_size=16,k=k)
x = concatenate(dense_lst2)
x = transitionLayer(x)

dense_lst3 = DenseBlock(x,no_layers=24,stage=3,feature_size=8,k=k)
x = concatenate(dense_lst3)
x = transitionLayer(x)

x = GlobalAveragePooling2D()(x)
#x = Flatten()(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.summary()

opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs=400

# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False)  # randomly flip images

model.fit(x_train, y_train,
                         batch_size=128,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

# model.fit_generator(datagen.flow(x_train, y_train,
#                                      batch_size=128),
#                         steps_per_epoch=int(np.ceil(x_train.shape[0] / float(128))),
#                         epochs=epochs,
#                         validation_data=(x_test, y_test),
#                         workers=8)
