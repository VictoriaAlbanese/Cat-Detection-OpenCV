################################################################
#
# Filename: test_nn.py
#
# Purpose: Contains code which loads a previously trained 
# neural network and tests it on unseen testing data, running
# some metrics on the results
#
################################################################
 
from sklearn.metrics import classification_report
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

import metric_functions
import nn_functions as nn
from feature_extractors import baseline_extractor
from feature_extractors import threshold_extractor
from feature_extractors import edge_extractor
from feature_extractors import DOG_extractor
from feature_extractors import HOG_extractor

CLASSES = ["cat", "dog"]

################################################################

# load the network
print("[INFO] loading model from {}".format(nn.MODEL_SAVE_LOCATION))
model = load_model(nn.MODEL_SAVE_LOCATION)

# load the images
print("[INFO] loading images in {}".format(nn.TEST_DATA_LOCATION))
X, y = nn.extract_image_data(nn.TEST_DATA_LOCATION, nn.EXTRACTOR, debug=True)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(X, y,	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

# extract the predicted and actual labels
y_predict = [model.predict(X)[index].argmax(axis=0) for (index, value) in enumerate(y)]
y_actual = [0 if value[0] == 1.0 else 1 for (index, value) in enumerate(y)]
print(y_actual, y_predict)

'''
TODO: Convert this to matplotlib stuff
# draw the class and probability on the test image and display it to our screen
for (i, label) in enumerate(labels) :
	label = "{}: {:.2f}%".format(CLASSES[prediction], probs[prediction] * 100)
	cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
'''

# print classification report 
# recall of positive class is sensitivity
# recall of negative class is specificity
print(classification_report(y_actual, y_predict))

# plot roc curve & confusion matrix
metric_functions.plot_roc_curve(y_actual, y_predict)
metric_functions.plot_confusion_matrix(y_actual, y_predict)

################################################################