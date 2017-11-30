from keras.layers import Conv2D, BatchNormalization, Dense, Dropout, merge, ZeroPadding2D
from keras.layers import concatenate, GlobalAveragePooling2D,MaxPooling2D, Input, Flatten,AveragePooling2D
from keras.activations import relu
from keras import losses
from keras.models import Model
from keras.layers import Activation
from keras.regularizers import l2
import keras
import os, sys, wget
import tarfile

from keras.datasets import cifar10
from keras.optimizers import Adam, SGD,RMSprop

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import ReduceLROnPlateau
# load in the CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def DenseBlock(x,no_layers,nb_filters,stage,grow_rt):
    concat_layers = [x]
    for i in range(0,no_layers):
        conv_b = ConvBlock(x,name_lyr='concat'+str(stage)+str(np.random.random_integers(0,1000000,1)[0]),nb_filters=nb_filters)
        x = concatenate([x, conv_b])
        nb_filters += grow_rt
        print('Adding denselayer {}'.format(i))
        
    return x, nb_filters


def transitionLayer(x,nb_filters):
    x = BatchNormalization(epsilon = 1.1e-5)(x)
    x = Conv2D(int(nb_filters * 0.5),(1,1),padding='same', dilation_rate = 1,kernel_initializer='he_uniform',use_bias=False)(x)
    x = Dropout(p=0.2)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((2,2),strides=2)(x)
    return x

def ConvBlock(x,name_lyr,nb_filters):
    x = BatchNormalization()(x)
    x = Conv2D(int(nb_filters*0.5),(1,1),padding='same', dilation_rate = 1,kernel_initializer='he_uniform',activation = 'relu',W_regularizer=l2(1E-4),use_bias=False)(x)
    x = Dropout(p=0.2)(x)
    x = BatchNormalization()(x)
    #x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filters,(3,3),padding='same', dilation_rate = 1,kernel_initializer='he_uniform',activation = 'relu',W_regularizer=l2(1E-4),use_bias=False)(x)
    return x

nb_filters = 24
grow_rt = 12
inputs = Input(shape=x_train.shape[1:])

x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(inputs)
x = Conv2D(int(nb_filters),(3,3),padding='same', dilation_rate = 1,name='init_conv',kernel_initializer='he_uniform',activation = 'relu',W_regularizer=l2(1E-4),use_bias=False)(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2,2),strides=2)(x)

x, nb_filters = DenseBlock(x,no_layers=16,stage=1,nb_filters=nb_filters,grow_rt=grow_rt)
x = Dropout(p=0.2)(x)
x = transitionLayer(x,nb_filters=nb_filters)

x, nb_filters = DenseBlock(x,no_layers=16,stage=2,nb_filters=nb_filters,grow_rt=grow_rt)
x = Dropout(p=0.2)(x)
x = transitionLayer(x,nb_filters=nb_filters)

x, nb_filters = DenseBlock(x,no_layers=16,stage=3,nb_filters=nb_filters,grow_rt=grow_rt)
x = Dropout(p=0.2)(x)

x = BatchNormalization(epsilon=1.1e-5)(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)

predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

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

model.fit(x_train, y_train,
                         batch_size=64,
                        epochs=epochs,
                        validation_data=(x_test, y_test),callbacks=[reduce_lr])

# model.fit_generator(datagen.flow(x_train, y_train,
#                                      batch_size=128),
#                         steps_per_epoch=int(np.ceil(x_train.shape[0] / float(128))),
#                         epochs=epochs,
#                         validation_data=(x_test, y_test),
#                         workers=8)
