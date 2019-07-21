import glob
import numpy as np
import cv2
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from DataProcessing import processing
from common.Constants import *
import sys
import os
from optparse import OptionParser
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import pickle


model_path = './cnn6_model.hdf5'
train_history = "./cnn6_history.pickle"

def create_model(input_shape):
    """
    CNN Keras model with 4 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt

def create_model_six_conv(input_shape):
    """
    CNN Keras model with 6 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt    

def load_model_from_checkpoint(weights_path, six_conv=False, input_shape=(pic_size,pic_size,3)):
    print(input_shape)
    if six_conv:
        model, opt = create_model_six_conv(input_shape)
    else:
        model, opt = create_model(input_shape)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model

def lr_schedule(epoch):
    lr = 0.01
    return lr*(0.1**int(epoch/10))

def run_model():
        X_train, X_test, y_train, y_test = processing.load_data_split()
        model, opt = create_model_six_conv(X_train.shape[1:])
        model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

        hist = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test),
                shuffle=True)

        #   Evaluate the model with the test data to get the scores on "real" data.
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save(model_path)
        with open(train_history, 'wb') as file_pi:
                pickle.dump(hist.history,file_pi)

        # #   Plot data to see relationships in training and validation data
        # import numpy as np
        # import matplotlib.pyplot as plt
        # epoch_list = list(range(1, len(hist.history['acc']) + 1))  # values for x axis [1, 2, ..., # of epochs]
        # plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
        # plt.legend(('Training Accuracy', 'Validation Accuracy'))
        # plt.show()


if __name__ == "__main__":
        parser = OptionParser()
        parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
        (options, args) = parser.parse_args()
        # options.train_path = '/Users/ravirane/Desktop/GMU/DAEN690/CNN/'
        if not options.train_path:   # if filename is not given
                parser.error('Error: path to training data must be specified. Pass --path to command line')
        os.chdir(options.train_path)
        run_model()
