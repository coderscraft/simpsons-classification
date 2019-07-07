import cv2
import numpy as np
import random
import pprint
import os.path
import csv


map_characters = {1: 'abraham_grampa_simpson', 2: 'apu_nahasapeemapetilon', 3: 'bart_simpson', 
        4: 'charles_montgomery_burns', 5: 'chief_wiggum', 6: 'comic_book_guy', 7: 'edna_krabappel', 
        8: 'homer_simpson', 9: 'kent_brockman', 10: 'krusty_the_clown', 11: 'lisa_simpson', 
        12: 'marge_simpson', 13: 'milhouse_van_houten', 14: 'moe_szyslak', 
        15: 'ned_flanders', 16: 'nelson_muntz', 17: 'principal_skinner', 18: 'sideshow_bob', 19: 'agnes_skinner', 20: 'barney_gumble'}

map_inrs = {'abraham_grampa_simpson': 1, 'apu_nahasapeemapetilon': 2, 'bart_simpson': 3, 
        'charles_montgomery_burns': 4, 'chief_wiggum': 5, 'comic_book_guy': 6, 'edna_krabappel': 7, 
        'homer_simpson': 8, 'kent_brockman': 9, 'krusty_the_clown': 10, 'lisa_simpson': 11, 
        'marge_simpson': 12, 'milhouse_van_houten': 13, 'moe_szyslak': 14, 
        'ned_flanders': 15, 'nelson_muntz': 16, 'principal_skinner': 17, 'sideshow_bob': 18, 'agnes_skinner': 19, 'barney_gumble': 20}

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




