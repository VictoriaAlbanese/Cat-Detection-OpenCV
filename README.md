# Cat-Detection-OpenCV
This repository contains our Computer Vision final project where we detect cat faces utilizing OpenCV.

-------------------

We got our dataset from here: https://www.kaggle.com/c/dogs-vs-cats/data

Our jumping off point is this tutorial: https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/

Our contribution is adding better feature extractors to improve the overall performance of the algorithm.  Currently, the image is just resized, flattened into a 1D array, and sent into the neural net.  We believe that this could be optimized by developing better feature vectors to feed to the program (especially since the image is distorted with the current method, which surely skews the results).
