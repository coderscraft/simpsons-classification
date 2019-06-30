import cv2
import numpy as np
import random
import pprint
import os.path

def get_data(input_path):
    # os.chdir(input_path)
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    anotfile = 'annotation.txt'
    with open(anotfile,'r') as f:
        print('Parsing annotation files')
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}
                # filename = os.path.join(input_path,filename)
                # print('--', os.path.abspath('.'))
                img = cv2.imread(filename)
                (rows,cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0,6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
        # print('>>>>',all_imgs)
        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])
        
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
        random.shuffle(all_data)
        print('Training images per class ({} classes) :'.format(len(classes_count)))
        pprint.pprint(classes_count)
        # print('<<<<',all_data)
        # print('888-',class_mapping)
        return all_data, classes_count, class_mapping


