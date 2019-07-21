from keras.models import load_model
import numpy as np
import cv2
from common.Constants import *
pic_size = 64
model = load_model('/Users/ravirane/Desktop/GMU/DAEN690/CNN/cnn6_model.hdf5')
a = cv2.imread("/Users/ravirane/Desktop/GMU/DAEN690/CNN/characters/comic_book_guy/pic_0003.jpg", 1)
pic = cv2.resize(a, (pic_size,pic_size))
print(pic)
pic1 = pic.reshape(1, pic_size, pic_size,3)
a = model.predict_proba(pic1)[0]
print(a)
print('Prediction')
print(map_characters[np.argmax(a)].replace('_',' ').title())