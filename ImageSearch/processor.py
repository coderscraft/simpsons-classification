import glob
import os
import pickle
from PIL import Image
import sys
from optparse import OptionParser
from featureExtractor import FeatureExtractor


ROOT_DIR = os.path.abspath("./ImageSearch")
sys.path.append(ROOT_DIR)

parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")

(options, args) = parser.parse_args()
# options.train_path = '/Users/ravirane/Desktop/GMU/DAEN690/search/'
# options.train_path = '/Users/ravirane/Desktop/GMU/DAEN690/data/the-simpsons-characters-dataset/'
os.chdir(options.train_path)
if not options.train_path:   # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')

if not os.path.exists('./static'):
    os.makedirs('./static')

fe = FeatureExtractor()
featureMap = {}
for img_path in sorted(glob.glob(options.train_path + '/characters/*/*.jpg')):
    print(img_path)
    fileName = os.path.splitext(os.path.basename(img_path))[0]
    img = Image.open(img_path)  # PIL image
    feature = fe.extract(img)
    featureMap[fileName] = feature
feature_path = './static/features' + '.pkl'
pickle.dump(featureMap, open(feature_path, 'wb'))