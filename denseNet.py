from keras.layers import Conv2D, BatchNormalization, Dense, Dropout, merge, ZeroPadding2D
from keras.layers import Concatenate, GlobalAveragePooling2D,MaxPooling2D, Input, Flatten,AveragePooling2D
from keras.activations import relu
from keras import losses
from keras.models import Model
from keras.layers import Activation
from keras.regularizers import l2
import keras
import sys, wget
import tarfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from keras.datasets import cifar10
from keras.optimizers import Adam, SGD,RMSprop

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import ReduceLROnPlateau
# load in the CIFAR10 dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

X = np.vstack((X_train, X_test))
for i in range(3):
    mean = np.mean(X[:, :, :, i])
    std = np.std(X[:, :, :, i])
    X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
    X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

def DenseBlock(x,no_layers,nb_filters,grow_rt):
    concat_layers = [x]
    for i in range(no_layers):
        conv_b = ConvBlock(x,grow_rt)
        concat_layers.append(conv_b)
        x = Concatenate(axis=-1)(concat_layers)
        nb_filters += grow_rt
        print('Adding denselayer {}'.format(i))
        
    return x, nb_filters


def transitionLayer(x,nb_filters,compression):
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    inter_ch = int(nb_filters*compression)
    x = Conv2D(int(inter_ch),(1,1),padding='same', dilation_rate = 1,kernel_initializer='he_uniform',use_bias=False,kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(rate=0.2)(x)
    x = AveragePooling2D((2,2),strides=(2, 2))(x)
    return x

def ConvBlock(x,nb_filters):
    inter_ch = int(nb_filters*4)
    x = Conv2D(int(inter_ch), (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Conv2D(nb_filters,(3,3),padding='same', dilation_rate = 1,kernel_initializer='he_uniform',activation = 'relu',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
    x = Dropout(rate=0.2)(x)
    return x

nb_filters=24
grow_rt=12
weight_decay = 1E-4
compression = 0.5
depth = 100
no_layers = int(((depth-4)/3)/2)


inputs = Input(shape=X_train.shape[1:])

x = Conv2D(nb_filters,(3,3),padding='same', dilation_rate = 1,kernel_initializer='he_uniform',activation = 'relu',kernel_regularizer=l2(weight_decay),use_bias=False)(inputs)

x, nb_filters = DenseBlock(x,no_layers=no_layers,nb_filters=nb_filters,grow_rt=grow_rt)
x = transitionLayer(x,nb_filters=nb_filters,compression=compression)
nb_filters = int(nb_filters * compression)

x, nb_filters = DenseBlock(x,no_layers=no_layers,nb_filters=nb_filters,grow_rt=grow_rt)
x = transitionLayer(x,nb_filters=nb_filters,compression=compression)
nb_filters = int(nb_filters * compression)

x, nb_filters = DenseBlock(x,no_layers=no_layers,nb_filters=nb_filters,grow_rt=grow_rt)
x = BatchNormalization(gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)

x = Dense(10, activation='softmax',kernel_regularizer=l2(weight_decay),bias_regularizer=l2(weight_decay))(x)

model = Model(inputs=inputs, outputs=[x])

model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=20, min_lr=0.00001)


opt =  Adam(lr=0.1)
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

model.fit(X_train, Y_train,
                         batch_size=64,
                        epochs=epochs,
                        validation_data=(X_test, Y_test),callbacks=[reduce_lr])

# model.fit_generator(datagen.flow(x_train, y_train,
#                                      batch_size=128),
#                         steps_per_epoch=int(np.ceil(x_train.shape[0] / float(128))),
#                         epochs=epochs,
#                         validation_data=(x_test, y_test),
#                         workers=8)
