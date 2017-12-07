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
import os, math
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from keras.datasets import cifar10
from keras.optimizers import Adam, SGD,RMSprop

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
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
    return x, nb_filters


def transitionLayer(x,nb_filters):
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filters,(1,1),padding='same',dilation_rate = 2,kernel_initializer='he_uniform',use_bias=False,kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(p=0.2)(x)
    x = AveragePooling2D((2,2),strides=(2, 2))(x)
    return x

def ConvBlock(x,nb_filters):
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filters),(3,3),padding='same', dilation_rate = 2,kernel_initializer='he_uniform',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
    x = Dropout(p=0.2)(x)
    return x

nb_filters=16
grow_rt=12
weight_decay = 1E-4

inputs = Input(shape=X_train.shape[1:])

x = Conv2D(nb_filters,(3,3),padding='same', dilation_rate = 1, kernel_initializer='he_uniform',kernel_regularizer=l2(weight_decay),use_bias=False,name="initial_conv2D")(inputs)
x = BatchNormalization(gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
x = Activation('relu')(x)
x = Conv2D(nb_filters,(3,3),padding='same', dilation_rate = 2, kernel_initializer='he_uniform',kernel_regularizer=l2(weight_decay),use_bias=False,name="initial2_conv2D")(x)

x, nb_filters = DenseBlock(x,no_layers=12,nb_filters=nb_filters,grow_rt=grow_rt)
x = transitionLayer(x,nb_filters=nb_filters)

x, nb_filters = DenseBlock(x,no_layers=12,nb_filters=nb_filters,grow_rt=grow_rt)
x = transitionLayer(x,nb_filters=nb_filters)

x, nb_filters = DenseBlock(x,no_layers=12,nb_filters=nb_filters,grow_rt=grow_rt)
x = BatchNormalization(gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)

x = Dense(10, activation='softmax',kernel_regularizer=l2(weight_decay),bias_regularizer=l2(weight_decay))(x)

model = Model(inputs=inputs, outputs=[x])

model.summary()

def step_decay(epoch):
    initial_lrate = 0.1
    if epoch < 150: 
        lrate = 0.1
    if epoch == 150:
        lrate = initial_lrate / 10
    if epoch > 150 and epoch < 225:
        lrate = initial_lrate / 10 
    if epoch >= 225:
        lrate = initial_lrate / 100
    return float(lrate)

lrate = LearningRateScheduler(step_decay)


opt =  SGD(lr=0.1,momentum=0.9)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs=300

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
                        validation_data=(X_test, Y_test),callbacks=[lrate])
