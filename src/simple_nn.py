# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
from glob import glob
import numpy as np
import cv2
import os

from feature_extractors import intensity_extractor
from feature_extractors import DOG_extractor
from feature_extractors import HOG_extractor

# grab the list of images that we'll be describing
print("[INFO] describing images...")
data_path = os.getcwd() + "\\dataset\\training_data\\"
image_paths = [f for f in glob(data_path + '*') if '.jpg' in f]

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images and...
#   - convert the image to a feature vector + add it to the data list
#   - extract the label from the image name + add it to the labels list
# showing an update every 1000 images
for (i, im_path) in enumerate(image_paths) :
    image = cv2.imread(im_path, 0)
    label = im_path.split(os.path.sep)[-1].split(".")[0]
    labels.append(label)
    features = DOG_extractor(image)
    data.append(features)
    if i > 0 and i % 100 == 0 :
        print("[INFO] processed {}/{}".format(i, len(image_paths)))

# scale the input image pixels to the range [0, 1]
data = np.array(data) / 255.0

# encode the labels, converting them from strings to integers, 
# then transform the labels into vectors in the range [0, num_classes] 
# this generates a vector for each label where the index of the 
# label is set to `1` and all other entries to `0`
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(training_data, testing_data, training_labels, testing_labels) = train_test_split(
	data, labels, test_size=0.25, random_state=42)

# define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072, kernel_initializer="uniform", activation="relu", ))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(training_data, training_labels, epochs=50, batch_size=128, verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testing_data, testing_labels,	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
 
# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
output_path = os.getcwd() + "\\output\\trained_network.hdf5"
model.save(output_path)