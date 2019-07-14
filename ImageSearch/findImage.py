import os
import numpy as np
from PIL import Image
from featureExtractor import FeatureExtractor
import glob
import pickle

fe = FeatureExtractor()
img = Image.open('/Users/ravirane/Desktop/GMU/DAEN690/search/characters/abraham_grampa_simpson/pic_0000.jpg')
query = fe.extract(img)
# print('print', query)
with open('/Users/ravirane/Desktop/GMU/DAEN690/search/static/features.pkl', 'rb') as f_in:
    features = pickle.load(f_in)
# Iterating over values 
featureList = []
imgList = []
for imageName, feature in features.items(): 
    featureList.append(feature)
    imgList.append(imageName)
# print(imgList) 
dists = np.linalg.norm(featureList - query, axis=1)
# print('print', dists)
ids = np.argsort(dists)[:30] 
scores = [(dists[id], imgList[id]) for id in ids]
print('print', scores)
