import glob
import os
import pickle
from PIL import Image
import sys
from optparse import OptionParser
from featureExtractor import FeatureExtractor

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

ROOT_DIR = os.path.abspath("./ImageSearch")
sys.path.append(ROOT_DIR)

parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")

(options, args) = parser.parse_args()
options.train_path = '/Users/ravirane/Desktop/GMU/DAEN690/search/'
# options.train_path = '/Users/ravirane/Desktop/GMU/DAEN690/data/the-simpsons-characters-dataset/'
os.chdir(options.train_path)
if not options.train_path:   # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')

if not os.path.exists('./static'):
    os.makedirs('./static')

fe = FeatureExtractor()
featureMap = {}
for img_path in sorted(glob.glob(options.train_path + '/characters/*/*.jpg')):
    path = remove_prefix(img_path, options.train_path)
    img = Image.open(img_path)  # PIL image
    feature = fe.extract(img)
    featureMap[path] = feature
feature_path = './static/features' + '.pkl'
pickle.dump(featureMap, open(feature_path, 'wb'))