################################################################
#
# Filename: nn_functions.py
#
# Purpose: Contains a bunch of functions that are frequently
# used in a neural network for image classification
#
################################################################

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from skimage import io
from glob import glob
import numpy as np
import os
import feature_extractors as fe

LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 128

################################################################

# GET FULL PATH
# get the full path to the relative location specified
def get_full_path(relative_path) :
    return os.getcwd() + relative_path

# LOAD IMAGE PATHS FUNCTION
# this function returns a list of the locations of all 
# the images at the location pointed to by the given path
def load_image_paths(dataset_path) :
    image_paths = [i for i in glob(dataset_path + '*') if '.jpg' in i]
    np.random.shuffle(image_paths)
    return image_paths

# EXTRACT IMAGE FEATURES
# this function returns a list of extracted features from 
# all the images at the location pointed to by the given path
def extract_image_data(dataset_path, extractor, debug=False) :
    image_paths = load_image_paths(dataset_path)
    encoder = LabelEncoder()
    X = []
    y = []
    for (i, image_path) in enumerate(image_paths) :
        image = io.imread(image_path, True)
        X.append(extractor(image))
        y.append(image_path.split(os.path.sep)[-1].split(".")[0])
        if i > 0 and i % 1000 == 0 :
            print("[INFO] processed {}%".format(i / len(image_paths) * 100))
            if debug : break
    X = np.array(X)
    y = encoder.fit_transform(y)
    y = np_utils.to_categorical(y, 2)
    return (X, y)

# CRAFT MODEL FUNCTION
# define the architecture of the network
# where the size is the length of the feature vector
def craft_model(size) :
    model = Sequential()
    model.add(Dense(int(size / 4), input_dim=size, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(int(size / 8), activation="relu", kernel_initializer="uniform"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    return model

# TRAIN MODEL FUNCTION
# this function trains the model in 50 epochs
# printing out lots of useful information along the way
# and saving the finished model to a file for convenience
def train_model(model, X, y) :
    model.compile(loss="binary_crossentropy", optimizer=SGD(lr=LEARNING_RATE), metrics=["accuracy"])
    model.fit(X, y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    model.save(MODEL_SAVE_LOCATION)

################################################################

EXTRACTOR = fe.HOG_extractor
PIXELS_PER_CELL = "4x4"

TRAINING_DATA_LOCATION = get_full_path("\\dataset\\training_data\\")
TEST_DATA_LOCATION = get_full_path("\\dataset\\test_data\\")
if EXTRACTOR == fe.HOG_extractor:
    MODEL_SAVE_LOCATION = get_full_path("\\output\\trained_nns\\nn_{}_{}.hdf5".format(EXTRACTOR.__name__, PIXELS_PER_CELL))
else:
    MODEL_SAVE_LOCATION = get_full_path("\\output\\trained_nns\\nn_{}.hdf5".format(EXTRACTOR.__name__))

################################################################