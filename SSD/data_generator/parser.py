import cv2
import numpy as np
import random
import pprint
import os.path
import csv


map_characters = {1: 'abraham_grampa_simpson', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'homer_simpson', 5: 'lisa_simpson', 
        6: 'marge_simpson', 7: 'ned_flanders', 8: 'principal_skinner'}

map_inrs = {'abraham_grampa_simpson': 1, 'bart_simpson': 2, 'charles_montgomery_burns': 3,
        'homer_simpson': 4, 'lisa_simpson': 5, 'marge_simpson': 6, 'ned_flanders': 7, 'principal_skinner': 8}

def buildData():
    train_list = []
    validation = []
    minW = 9999999
    minH = 9999999 
    anotfile = 'annotation.txt'
    with open(anotfile,'r') as f:
        print('Parsing annotation files')
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split
            # print(filename,x1,y1,x2,y2,class_name)
            data = {}
            data['frame'] = filename
            data['xmin'] = x1
            data['xmax'] = x2
            data['ymin'] = y1
            data['ymax'] = y2
            data['class_id'] = map_inrs[class_name]
            
            img = cv2.imread(filename)
            (h,w) = img.shape[:2]
            # print(h, w)
            if w < minW:
                minW = w
            if h < minH:
                minH = h

            if np.random.randint(0,5) > 0:
                train_list.extend([data])
            else:
                validation.extend([data])
    keys = train_list[0].keys()
    with open('train_list.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(train_list)
    with open('valid_list.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(validation)
    return minW, minH




