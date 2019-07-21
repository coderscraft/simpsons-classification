# export PYTHONPATH="/Users/ravirane/python/simpsons-classification/"

from optparse import OptionParser
from DataProcessing import processing
from simpsons import load_model_from_checkpoint
from keras.models import load_model
import numpy as np
import sklearn
from common.Constants import *
import os

def run_prediction():
    X_train, X_test, y_train, y_test = processing.load_data_split()
    # model = load_model_from_checkpoint('/Users/ravirane/Desktop/GMU/DAEN690/data/the-simpsons-characters-dataset/weights.best.hdf5', six_conv=True)
    model = load_model('/Users/ravirane/Desktop/GMU/DAEN690/CNN/cnn6_model.hdf5')
    y_pred = model.predict(X_test)
    print(y_pred)
    print(y_test)
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], 
                                                  np.argmax(y_pred, axis=1),
                                                  labels=labels,
                                                  target_names=list(map_characters.values())), sep='')   
    print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], 
                                                  np.argmax(y_pred, axis=1)), sep='')                                                

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
    (options, args) = parser.parse_args()
    options.train_path = '/Users/ravirane/Desktop/GMU/DAEN690/CNN/'
    if not options.train_path:   # if filename is not given
            parser.error('Error: path to training data must be specified. Pass --path to command line')
    os.chdir(options.train_path)
    run_prediction()    

