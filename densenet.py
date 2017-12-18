# sys args are epochs (300), no_layers (12), growth_rate (12), nb_filters (16), nb_classes (10 - CIFAR10), data_augmentation (False)

# no_layers = int(((depth-4)/3)/2) < for compression else no_layers = int((depth-4)/3)

# DenseNet k =12 Depth = 40 - sys.argv = [300, 12, 12, 16, 10, False, 3, 1]

# DenseNet-BC k =12, Depth = 100 - sys.argv = [300, 16, 12, 24, 10, False, 3, 0.5]

import numpy as np

np.random.seed(42)

from keras.layers import Conv2D, BatchNormalization, Dense, Dropout
from keras.layers import merge, ZeroPadding2D, UpSampling2D
from keras.layers import Concatenate, GlobalAveragePooling2D,MaxPooling2D, concatenate
from keras.layers import Input, Flatten,AveragePooling2D
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
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler


# setup the parameters - user specified
# nb_filters=16
# grow_rt=12
# weight_decay = 1E-4

class DenseNet():
	def load_cifar(self):
		""" Load the CIFAR10 dataset, normalise it, and return it as a tuple """
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
		return (X_train, Y_train), (X_test, Y_test)

	def ConvBlock(self,x,nb_filters,drop_rate,dilate_rate,weight_decay,compression):
		""" create a standard convolutional block, BN - rel - Conv2D (3x3) - Dropout """
		if compression != 1:
			inter_ch = int(nb_filters*4)

			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
			beta_regularizer=l2(weight_decay))(x)

			x = Activation('relu')(x)

			x = Conv2D(int(inter_ch), (1, 1), kernel_initializer='he_uniform',
			padding='same', use_bias=False,kernel_regularizer=l2(weight_decay))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
			beta_regularizer=l2(weight_decay))(x)

			x = Activation('relu')(x)

			x = Conv2D(int(nb_filters),(3,3),padding='same',
			dilation_rate = dilate_rate,kernel_initializer='he_uniform',
			kernel_regularizer=l2(weight_decay),use_bias=False)(x)
			x = Dropout(rate=drop_rate)(x)
		else:
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
			beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			x = Conv2D(int(nb_filters),(3,3),padding='same',
			dilation_rate = dilate_rate,kernel_initializer='he_uniform',
			kernel_regularizer=l2(weight_decay),use_bias=False)(x)
			x = Dropout(rate=drop_rate)(x)
		return x

	def transitionLayer(self,x,nb_filters,dilate_rate,weight_decay,drop_rate,compression):
		""" Create a transition layer BN - relu - Conv2d (1x1), Dropout """
		x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		if compression != 1:
			inter_ch = int(nb_filters*compression)
			x = Conv2D(inter_ch,(1,1),padding='same',dilation_rate = dilate_rate,
			kernel_initializer='he_uniform',use_bias=False,kernel_regularizer=l2(weight_decay))(x)
		else:
			x = Conv2D(nb_filters,(1,1),padding='same',dilation_rate = dilate_rate,
			kernel_initializer='he_uniform',use_bias=False,kernel_regularizer=l2(weight_decay))(x)
		x = Dropout(rate=drop_rate)(x)
		x = AveragePooling2D((2,2),strides=(2, 2))(x)
		return x

	def DenseBlock(self,x,no_layers,nb_filters,grow_rt,drop_rate,dilate_rate,weight_decay,compression):
		""" Create a loop defining the number of conv and transition blocks to add to a denseblock """
		concat_layers = [x]
		for i in range(no_layers):
			conv_b = self.ConvBlock(x,grow_rt,drop_rate,dilate_rate,weight_decay,compression=compression)
			concat_layers.append(conv_b)
			x = Concatenate(axis=-1)(concat_layers)
			nb_filters += grow_rt
		return x, nb_filters

	def Upsampling(self,x,no_layers,nb_filters,grow_rt,drop_rate,dilate_rate,weight_decay,compression):
		x = BatchNormalization(gamma_regularizer=l2(weight_decay),
		beta_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = UpSampling2D((2,2))(x)
		x = Conv2D(int(nb_filters),(3,3),padding='same',
		dilation_rate = dilate_rate,kernel_initializer='he_uniform',
		kernel_regularizer=l2(weight_decay),use_bias=False)(x)
		x = BatchNormalization(gamma_regularizer=l2(weight_decay),
		beta_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = UpSampling2D((2,2))(x)
		return x

	def densemodel(self,no_layers,dilate_rate,grow_rt,nb_filters,nb_classes,
	weight_decay,drop_rate,nb_blocks,compression,upsampling):
		""" Create a model with 3 DenseBlocks, starts with an initial convolution (3x3)"""
		inputs = Input(shape=X_train.shape[1:])

		#initial convolution
		x_start = Conv2D(nb_filters,(3,3),padding='same', dilation_rate = dilate_rate,
			kernel_initializer='he_uniform',kernel_regularizer=l2(weight_decay),use_bias=False,name="initial_conv2D")(inputs)

		#create denseblocks
		for i in range(0,nb_blocks-1):
			if i == 0:
				x, nb_filters = self.DenseBlock(x=x_start,no_layers=no_layers,
				nb_filters=nb_filters,grow_rt=grow_rt,
				drop_rate=drop_rate, dilate_rate = dilate_rate,
				weight_decay = weight_decay,compression = compression)

				x = self.transitionLayer(x,nb_filters=nb_filters,
				dilate_rate =dilate_rate, weight_decay = weight_decay,
				drop_rate = drop_rate, compression = compression)
			else:
				x, nb_filters = self.DenseBlock(x,no_layers=no_layers,
				nb_filters=nb_filters,grow_rt=grow_rt,
				drop_rate=drop_rate, dilate_rate = dilate_rate,
				weight_decay = weight_decay,compression = compression)

				x = self.transitionLayer(x,nb_filters=nb_filters,
				dilate_rate =dilate_rate, weight_decay = weight_decay,
				drop_rate = drop_rate, compression = compression)

		# finish with denseblock without transition - BC-relu-GAP-prediction
		x, nb_filters = self.DenseBlock(x,no_layers=no_layers,
		nb_filters=nb_filters,grow_rt=grow_rt, drop_rate=drop_rate,
		dilate_rate = dilate_rate, weight_decay = weight_decay,compression = compression)

		if upsampling == '1':
			x_end = self.Upsampling(x,no_layers,nb_filters,grow_rt,drop_rate,dilate_rate,
			weight_decay,compression)
			x = concatenate([x_start,x_end])

		x = BatchNormalization(gamma_regularizer=l2(weight_decay),
		beta_regularizer=l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = GlobalAveragePooling2D()(x)
		x = Dense(nb_classes, activation='softmax',
		kernel_regularizer=l2(weight_decay),bias_regularizer=l2(weight_decay))(x)
		model = Model(inputs=inputs, outputs=[x])
		model.summary()
		return(model)

def step_decay(epoch):
	initial_lrate = 0.1
	lrate = 0.1
	if epoch >= 50 and epoch < 150:
		lrate = initial_lrate / 10
	if epoch >= 150 and epoch < 225:
		lrate = initial_lrate / 100
	if epoch >= 225:
		lrate = initial_lrate / 1000
	return float(lrate)

lrate = LearningRateScheduler(step_decay)

if __name__ == "__main__":
	epochs=int(sys.argv[1])
	init = DenseNet()
	(X_train, Y_train), (X_test, Y_test) = init.load_cifar()

	# initialise the densemodel from DenseNet class
	model = init.densemodel(no_layers=int(sys.argv[2]),dilate_rate=1,
	grow_rt=int(sys.argv[3]),nb_filters=int(sys.argv[4]),
	nb_classes=int(sys.argv[5]),weight_decay=1E-4,drop_rate=0.2,
	nb_blocks=int(sys.argv[7]),compression = float(sys.argv[8]),
	upsampling = sys.argv[9])

	opt =  SGD(lr=0.1,momentum=0.9)
	model.compile(optimizer=opt,
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
	data_aug=sys.argv[6]

	if data_aug == '1':
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False)  # randomly flip images

		model.fit_generator(datagen.flow(X_train, Y_train,
									 batch_size=64),
						steps_per_epoch=782,epochs=epochs,
						validation_data=(X_test, Y_test),
						workers=4,callbacks=[lrate])
	else:
		model.fit(X_train, Y_train,
						 batch_size=64,
						epochs=epochs,
						validation_data=(X_test, Y_test),callbacks=[lrate])
