import cv2
import os
from glob import glob
import numpy as np

dataPath = os.getcwd() + "\\cats_bigger_than_64x64\\"
images = [f for f in glob(dataPath + '*') if '.jpg' in f]



average_image = cv2.imread(images[0], 0)
for i in range(50) :
    alpha = float(i / (i + 1))
    beta = float(1 - alpha)
    next_image = cv2.imread(images[i + 1], 0)
    average_image = cv2.addWeighted(average_image, alpha, next_image, beta, 0)

cv2.imwrite('src/sift_imgs/average_cat.jpg'.format(i), average_image)
cv2.imshow("mean image", average_image)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(average_image, None)
img = cv2.drawKeypoints(average_image, kp, average_image)

cv2.imwrite('src/sift_imgs/average_sift.jpg', img)
cv2.imshow("sift mean image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()