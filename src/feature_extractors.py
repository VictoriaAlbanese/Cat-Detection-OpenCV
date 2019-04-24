from skimage import filters
import numpy as np
import cv2
from skimage.feature import hog
from skimage import io

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
# finds a difference of gaussians for each image and then
# extracts the feature vector from the image by resizing & flattening
def DOG_extractor(image, size=(32, 32)) :
    for i in range(1, 3) :
        g1 = filters.gaussian(image, 1.6 * 2**i)
        g2 = filters.gaussian(image, 2**i)
        dog = g1 - g2
        #cv2.imshow("Image", dog)
        #cv2.waitKey(0)
    feature_vector = cv2.resize(dog, size).flatten()
    return intensity_extractor(dog)

# HOG FEATURE EXTRACTOR FUNCTION
# finds a histogram of oriented gradients for each image 
# and uses that as the feature vector for the image 
def HOG_extractor(image, size=(32,32)):
    image = cv2.resize(image, size)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # tested with (2,2) [accuracy = 49%] and (4,4) [accuracy = 51%] pixels to almost no improvement
    hog_arr = hog(image)
    # hog_arr, hog_image = hog(image, pixels_per_cell=(4,4), visualize=True)
    # cv2.imshow("HOG", hog_image)
    # cv2.waitKey(0)
    return hog_arr

#image = io.imread("C:/Users/Hannah/Documents/computervisionfinal/Cat-Detection-OpenCV/dataset/training_data/cat.0.jpg", True)
#image = cv2.resize(image, (32,32))
#v = image_to_feature_vector(image)
#v2 = image_to_feature_vector(image, 'hog')
#print(len(v))
#print(len(v2))
