import cv2


# IMAGE TO FEATURE VECTOR 
#   image : the image to convert
#    size : the new size of the image
# converts an image to a feature vector
def image_to_feature_vector(image, size=(32, 32)):

    # this version of the function resizes the image into 
    # a 32x32 pixel image with 3 color channels, and then flattens
    # it into a 1D list of size 32x32x3 = 3072D feature vector

	return cv2.resize(image, size).flatten()