from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np
import tensorflow as tf

import glob
import numpy as np
import cv2
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import sys


class FeatureExtractor:

    def __init__(self):
        input_shape = (224, 224, 3)
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
        # model.add(Dense(num_classes))
        # model.add(Activation('softmax'))
        # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        self.model = model
        self.opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        self.graph = tf.get_default_graph()

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        with self.graph.as_default():
            feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
            return feature / np.linalg.norm(feature)  # Normalize