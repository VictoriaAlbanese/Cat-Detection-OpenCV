################################################################
#
# Filename: train_nn.py
#
# Purpose: Contains code which trains a fully connected neural 
# network with some training data that needs to be classified
#
################################################################

import nn_functions as nn
from feature_extractors import baseline_extractor
from feature_extractors import threshold_extractor
from feature_extractors import edge_extractor
from feature_extractors import DOG_extractor
from feature_extractors import HOG_extractor

################################################################

# load the images
print("[INFO] loading images in {}".format(nn.TRAINING_DATA_LOCATION))
X, y = nn.extract_image_data(nn.TRAINING_DATA_LOCATION, nn.EXTRACTOR, debug=True)

# define the architecture of the network
print("[INFO] crafting model...")
size = 1024
if nn.EXTRACTOR is DOG_extractor : size = 3072
if nn.EXTRACTOR is HOG_extractor : size = 324
model = nn.craft_model(size)

# train the model & save to file
print("[INFO] training model...")
nn.train_model(model, X, y)
print("[INFO] saving architecture & weights to {}".format(nn.MODEL_SAVE_LOCATION))

################################################################