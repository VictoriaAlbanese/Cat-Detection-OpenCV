import cv2
import os
from glob import glob
import numpy as np
from sklearn import svm


dataPath = os.getcwd() + "\\cats_processed_1000\\"
imgs = [f for f in glob(dataPath + '*') if '.jpg' in f]
i = 1
for f in imgs:
    image = cv2.imread(f, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    kp, des = sift.compute(gray, kp)

    img = cv2.drawKeypoints(gray, kp, image)
    cv2.imwrite('src/sift_imgs/sift_keypoints{0}.jpg'.format(i), img)
    i += 1
    if i == 50:
        break

print(des)
# k-means clustering
# into svm
