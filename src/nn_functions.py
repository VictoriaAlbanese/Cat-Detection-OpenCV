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
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from glob import glob
import cv2
import os

################################################################

# GET FULL PATH
# get the full path to the relative location specified
def get_full_path(relative_path) :
    return os.getcwd() + relative_path

# LOAD IMAGE PATHS FUNCTION
# this function returns a list of the locations of all 
# the images at the location pointed to by the given path
def load_image_paths(dataset_path) :
    image_paths = [image_path for image_path in glob(dataset_path + '*') if '.jpg' in image_path]
    return image_paths

# LOAD IMAGES FUNCTION
# this function returns a list of all the images 
# at the location pointed to by the given path
def load_image_paths(dataset_path) :
    image_paths = [image_path for image_path in glob(dataset_path + '*') if '.jpg' in image_path]
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images

# EXTRACT IMAGE FEATURES
# this function returns a list of extracted features from 
# all the images at the location pointed to by the given path
def extract_image_data(dataset_path, extractor) :
    image_paths = load_image_paths(dataset_path)
    encoder = LabelEncoder()
    X = []
    y = []
    for (i, image_path) in enumerate(image_paths) :
        y.append(im_path.split(+os.path.sep)[-1].split(".")[0])
        image = cv2.imread(image_path)
        X.append(extractor(image))
        if i > 0 and i % 1000 == 0 :
            print("[INFO] processed {}/{}".format(i, len(images)))
    y = encoder.fit_transform(y)
    y = np_utils.to_categorical(y, 2)
    return (X, y)

# SPLIT DATA FUNCTION
# this function divvys up the data between the 
# training dataset and the testing dataset, allocating 
# testing_size to testing and the rest to training
def split_data(X, y, testing_size) :
    return train_test_split(X, y, test_size=testing_size, random_state=42)

# CRAFT MODEL FUNCTION
# define the architecture of the network
# where the size is the length of the feature vector
def craft_model(size) :
    model = Sequential()
    model.add(Dense(size / 4, input_dim=size, kernel_initializer="uniform", activation="relu", ))
    model.add(Dense(size / 8, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    return model

# TRAIN MODEL FUNCTION
# this function trains the model in 50 epochs
# printing out lots of useful information along the way
# and saving the finished model to a file for convenience
def train_model(model, X, y, output_file_description) :

    print("[INFO] compiling model...")
    sgd = SGD(lr=0.01)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
    model.fit(training_data, training_labels, epochs=50, batch_size=128, verbose=1)

    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(testing_data, testing_labels,	batch_size=128, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
 
    print("[INFO] dumping architecture and weights to file...")
    output_path = get_full_path("\\output\\nn_" + output_file_description + ".hdf5")
    model.save(output_path)

    return output_path

# MAKE PREDICTIONS FUNCTION
# this model makes predictions on the first n_predict
# number of given images using the given model
def make_predictions(model, images, n_predict) :

    X = nn.extract_image_features(data_path, EXTRACTOR)
    y = []

    if n_predict < len(images) : N = n_predict
    else : N = len(images)

    for i in range(N) :
        
        probabilities = model.predict(features)[0]
        prediction = probs.argmax(axis=0)
        y.append(prediction)

        label = "{}: {:.2f}%".format(CLASSES[prediction], probs[prediction] * 100)
        cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    

################################################################