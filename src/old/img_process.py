import cv2
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data, exposure
from skimage.color import rgb2gray
from skimage.filters import roberts, sobel
from skimage.feature import hog


dataPath = os.getcwd() + "\\dataset\\training_data\\"
imgs = [f for f in glob(dataPath + '*') if '.jpg' in f]
i = 1
data = []
for f in imgs:
    # converts numpy 2d array, in grayscale
    image = io.imread(f, True)
    # gets HOG arrays and displayable image
    hog_array, hog_image = hog(image, visualise=True)
    cv2.imshow("HOG", hog_image)
    cv2.waitKey(0)
    #fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
      #                 figsize=(16, 8))
    #ax[0].imshow(hog_image, cmap=plt.cm.gray)
    #ax[0].set_title('HOG ME')
    
    # experimenting with sobel edge detection...
    edge_sobel = sobel(image)
    #ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
    #ax[1].set_title('Sobel Edge Detection')
    # uncomment plt stuff to view pretty pictures:
    #plt.show()

    data.append(hog_array)
   
    i += 1
    if i == 10:
        break

from sklearn import svm
labels = [1 for j in range(i)]
labels = np.array(labels)
data = np.array(data)
data_frame = np.hstack((data, labels))
print(data_frame)
np.random.shuffle(data_frame)
# percent training vs. test
train_p = 75
partition = int(len(data)*train_p/100)
x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel(), data_frame[partition:,-1:].ravel()
model = svm.SVC(gamma='scale')
model.fit(x_train, y_train)
model.predict(x_test)

# pad arrays with 0s so they are normalized
# max_len = max((len(l), f) for f, l in enumerate(data))[0]
# print(max_len)
# print(len(data[0]))
# data = [np.pad(data, (0,max_len-len(x)), 'edge') for x in data]
# print(data[0])
# X = data[int(i*.75):]
# test = data[:int(i*.25)]
# y = [j for j in range(i)]
# model = svm.SVC(gamma='scale')
# model.fit(X, y)
# model.predict(test)

# k-means clustering
# into svm
