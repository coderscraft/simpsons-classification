import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import pickle
import sys

train_path = '/Users/ravirane/Desktop/GMU/DAEN690/project/'
os.chdir(train_path)
train_history = "ssd7_history.pickle"

history = {}
with open(train_history, 'rb') as f_in:
        history = pickle.load(f_in)

plt.figure(figsize=(20,12))
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend(loc='upper right', prop={'size': 24})
plt.show()