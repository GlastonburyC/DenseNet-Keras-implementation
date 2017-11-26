opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

model.compile(optimizer=opt,
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

