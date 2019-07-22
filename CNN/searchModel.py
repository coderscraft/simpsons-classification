from keras.models import Model, load_model
import numpy as np
import cv2
from common.Constants import *
import os
train_path = '/Users/ravirane/Desktop/GMU/DAEN690/CNN/'
os.chdir(train_path)
model_path = './cnn_feature_model.hdf5'
model = load_model('/Users/ravirane/Desktop/GMU/DAEN690/CNN/cnn6_model.hdf5')
model.summary()
feature_model = Model(inputs=model.input,
                                 outputs=model.get_layer('feature_dense').output)
feature_model.save(model_path)                                 
feature_model.summary()


# import pydot
# from keras.utils import plot_model
# from IPython.display import SVG
# import keras.utils.vis_utils 
# from keras.utils.vis_utils import model_to_dot
# plot_model(model, to_file='model.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))