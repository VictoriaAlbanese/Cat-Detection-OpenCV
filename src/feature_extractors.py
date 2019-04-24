import cv2
from skimage.feature import hog
from skimage import io

# IMAGE TO FEATURE VECTOR 
#   image : the image to convert
#    size : the new size of the image
# converts an image to a feature vector
def image_to_feature_vector(image, size=(32, 32)):
    # this version of the function resizes the image into 
    # a 32x32 pixel image with 3 color channels, and then flattens
    # it into a 1D list of size 32x32x3 = 3072D feature vector
    return cv2.resize(image, size).flatten()

def image_to_hog_vector(image, size=(32,32)):
    image = cv2.resize(image, size)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # tested with (2,2) [accuracy = ] and (4,4) [accuracy = 51%] pixels to almost no improvement
    hog_arr = hog(image)
    # hog_arr, hog_image = hog(image, pixels_per_cell=(4,4), visualize=True)
    # cv2.imshow("HOG", hog_image)
    # cv2.waitKey(0)
    return hog_arr

image = io.imread("C:/Users/Hannah/Documents/computervisionfinal/Cat-Detection-OpenCV/dataset/training_data/cat.0.jpg", True)
image = cv2.resize(image, (32,32))
v = image_to_feature_vector(image)
v2 = image_to_feature_vector(image, 'hog')
print(len(v))
print(len(v2))