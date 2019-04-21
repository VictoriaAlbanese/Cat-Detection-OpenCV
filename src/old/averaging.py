import cv2
import os
from glob import glob
import numpy as np

dataPath = os.getcwd() + "\\cats_processed_1000\\"
images = [f for f in glob(dataPath + '*') if '.jpg' in f]


average_image = cv2.imread(images[0], 0)
for i in range(50) :
    alpha = float(i / (i + 1))
    beta = float(1 - alpha)
    next_image = cv2.imread(images[i + 1], 0)
    average_image = cv2.addWeighted(average_image, alpha, next_image, beta, 0)

cv2.imwrite('src/sift_imgs/average_cat.jpg'.format(i), average_image)
cv2.imshow("mean image", average_image)


kp_image = average_image
for i in range(50) :
    next_image = cv2.imread(images[i + 1], 0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(next_image, None)
    #kp_image = cv2.drawKeypoints(kp_image, kp, kp_image)
    next_image = cv2.drawKeypoints(next_image, kp, next_image)
    cv2.imshow("next sift image", next_image)


cv2.imwrite('src/sift_imgs/average_sift.jpg', kp_image)
cv2.imshow("sift mean image", kp_image)


cv2.waitKey(0)
cv2.destroyAllWindows()