################################################################
#
# Filename: feature_extractors.py
#
# Purpose: Contains different feature extractors which filter
# the input image in different ways; the resulting feature 
# vectors are passed to the neural network for training
#
################################################################

import matplotlib.pyplot as plt
from skimage import transform
from skimage import filters
from skimage import feature
from skimage import io

################################################################

# RAW EXTRACTOR FUNCTION
# this function uses the pixels of the image as they are and then
# converts the image to a feature vector by resizing and flattening it
def baseline_extractor(image, size=(32, 32)):
    #io.imshow(image)
    #plt.show()
    feature_vector = transform.resize(image, size).flatten()
    feature_vector = [pixel / 255.0 for pixel in feature_vector]
    return feature_vector

# THRESHOLD EXTRACTOR FUNCTION
# this function converts the image into a binary image and then 
# converts the image to a feature vector by resizing and flattening it
def threshold_extractor(image, size=(32, 32)):
    thresh = filters.thresholding.threshold_otsu(image)
    binary_image = image > thresh
    #io.imshow(binary_image)
    #plt.show()
    feature_vector = transform.resize(binary_image, size).flatten()
    return feature_vector

# EDGE EXTRACTOR FUNCTION
# this function finds the sobel edges in the given image and then 
# converts the image to a feature vector by resizing and flattening it
def edge_extractor(image, size=(32, 32)):
    sobel_image = filters.sobel(image)
    #io.imshow(sobel_image)
    #plt.show()
    feature_vector = transform.resize(sobel_image, size).flatten()
    return feature_vector

# DOG FEATURE EXTRACTOR FUNCTION
# finds a difference of gaussians for each image and then
# extracts the feature vector from the image by resizing & flattening
def DOG_extractor(image, size=(32, 32)) :
    feature_vector = []
    for i in range(1, 4) :
        g1 = filters.gaussian(image, 1.6 * 2**i)
        g2 = filters.gaussian(image, 2**i)
        dog = g1 - g2
        feature_vector.extend(transform.resize(dog, size).flatten())
        #io.imshow(dog)
        #plt.show()
    return feature_vector

# HOG FEATURE EXTRACTOR FUNCTION
# finds a histogram of oriented gradients for each image 
# and uses that as the feature vector for the image 
def HOG_extractor(image, size=(32,32)) :
    image = cv2.resize(image, size)
    hog_arr, hog_image = hog(image, pixels_per_cell=(4,4), visualize=True)
    #io.imshow(hog_image)
    #plt.show()
    return hog_arr

################################################################