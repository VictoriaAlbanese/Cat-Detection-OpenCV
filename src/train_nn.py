# import the necessary packages
import feature_extractors
import nn_functions as nn
EXTRACTOR = feature_extractors.image_intensity

# load the images, extracting the features along the way
data_path = nn.get_full_path("\\dataset\\training_data\\")
print("[INFO] loading images in {}".format(data_path))
(X, y) = nn.extract_image_data(data_path, EXTRACTOR)

# partition the data into training data (75%) and testing data (25%)
print("[INFO] constructing training/testing split...")
(X_training, X_testing, y_training, y_testing) = split_data(X, y, 0.25)

# make the neural network model & train it 
model = nn.craft_model(len(y))
model_file = nn.train_model(model, X_training, y_training, EXTRACTOR.__name__)