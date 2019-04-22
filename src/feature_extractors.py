import cv2
from skimage.feature import hog

# IMAGE TO FEATURE VECTOR 
#   image : the image to convert
#    size : the new size of the image
# converts an image to a feature vector
def image_to_feature_vector(image, type='default', size=(32, 32)):

    if type is 'hog':
        image = cv2.resize(image, size)
        return hog(image).flatten()
    
    # this version of the function resizes the image into 
    # a 32x32 pixel image with 3 color channels, and then flattens
    # it into a 1D list of size 32x32x3 = 3072D feature vector
    return cv2.resize(image, size).flatten()

image = cv2.imread("C:/Users/Hannah/Documents/computervisionfinal/Cat-Detection-OpenCV/dataset/training_data/cat.0.jpg")
v = image_to_feature_vector(image)
v2 = image_to_feature_vector(image, 'hog')
print(len(v))
print(len(v2))