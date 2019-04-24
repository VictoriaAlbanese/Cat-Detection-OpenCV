from skimage import filters
import numpy as np
import cv2


# INTENSITY EXTRACTOR FUNCTION
# this version of the function resizes the image into 
# a 32x32 pixel image with 3 color channels, and then flattens
# it into a 1D list of size 32x32x3 = 3072D feature vector
# and scale the input image pixels to the range [0, 1]
def intensity_extractor(image, size=(32, 32)):
    ret, binary_image = cv2.threshold(image, 127, 255, cv.THRESH_BINARY)
    #cv2.imshow("BINARY IMAGE", binary_image)
    #cv2.waitKey(0)
    feature_vector = cv2.resize(binary_image, size).flatten()
    return feature_vector

# DOG FEATURE EXTRACTOR FUNCTION
# extracts the sift feature vector from the image
def DOG_extractor(image, size=(32, 32)) :
    for i in range(1, 3) :
        g1 = filters.gaussian(image, 1.6 * 2**i)
        g2 = filters.gaussian(image, 2**i)
        dog = g1 - g2
        #cv2.imshow("Image", dog)
        #cv2.waitKey(0)
    feature_vector = cv2.resize(dog, size).flatten()
    return intensity_extractor(dog)