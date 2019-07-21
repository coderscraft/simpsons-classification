import glob
import cv2
import numpy as np
from common.Constants import *
import keras
from sklearn.model_selection import train_test_split
import os.path

def load_pictures():
        pics = []
        labels = []
        # print(argo, os.getcwd())
        # path = os.path.abspath(os.path.join(os.getcwd(),'Data/SampleData')) + '/%s/*'
        # if argo:
        #         path = '/scratch/rrane/SampleData/%s/*'
        # print(path)
        for k, char in map_characters.items():
                pictures = [k for k in glob.glob('./characters/' + char + '/*')]       
                nb_pic = len(pictures)
                print(char)
                print(nb_pic)
                # if len(pictures) > 0:
                #     print(k)
                #     print(char)
                #     print(pictures)
                #     print(len(pictures))
                #     print('------')
                for pic in np.random.choice(pictures, nb_pic):
                        a = cv2.imread(pic, 1)
                        a = cv2.resize(a, (pic_size, pic_size))
                        # cv2.imshow('image',a)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        pics.append(a)
                        labels.append(k)
        return np.array(pics), np.array(labels)

def load_data_split():
        X, y = load_pictures()
        y = keras.utils.to_categorical(y, num_classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train = X_train.reshape(X_train.shape[0], pic_size, pic_size, 3)
        X_test = X_test.reshape(X_test.shape[0], pic_size, pic_size, 3)
        X_train = X_train / 255.
        X_test = X_test / 255.
        print("Train", X_train.shape, y_train.shape)
        print("Test", X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test

def load_data_split1(argo=False):
        X, y = load_pictures(argo)
        y = keras.utils.to_categorical(y, num_classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train = X_train.reshape(X_train.shape[0], pic_size, pic_size, 1)
        X_test = X_test.reshape(X_test.shape[0], pic_size, pic_size, 1)
        X_train = X_train / 255.
        X_test = X_test / 255.
        print("Train", X_train.shape, y_train.shape)
        print("Test", X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
# load_data_split()