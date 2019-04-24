# import the necessary packages
from __future__ import print_function
from keras.models import load_model
from imutils import paths
from glob import glob
import numpy as np
import argparse
import imutils
import cv2
import os
 
from feature_extractors import image_to_feature_vector

# initialize the class labels for the Kaggle dogs vs cats dataset
CLASSES = ["cat", "dog"]
 
# load the network
print("[INFO] loading network architecture and weights...")
model_path = os.getcwd() + "\\output\\trained_hog_network.hdf5"
model = load_model(model_path)

# load the images
data_path = os.getcwd() + "\\dataset\\test_data\\"
image_paths = [f for f in glob(data_path + '*') if '.jpg' in f]
print("[INFO] testing on images in {}".format(data_path))

# loop over our testing images
for im_path in image_paths :
	print("[INFO] classifying {}".format(im_path[im_path.rfind("/") + 1:]))
	image = cv2.imread(im_path)
	features = image_to_feature_vector(image, 'hog') / 255.0
	features = np.array([features])
    
    # classify the image using our extracted features and pre-trained
    # neural network
	probs = model.predict(features)[0]
	prediction = probs.argmax(axis=0)

    # draw the class and probability on the test image and display it
    # to our screen
	label = "{}: {:.2f}%".format(CLASSES[prediction], probs[prediction] * 100)
	cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
