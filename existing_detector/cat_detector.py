# import the necessary packages
import cv2
import os
from glob import glob

# construct the argument parse and parse the arguments
dataPath = os.getcwd() + "\\existing_detector\\"
haarPath = dataPath + "haarcascade_frontalcatface.xml"
imgs = [f for f in glob(dataPath + 'images\\*') if '.jpg' in f]

# iterate through all the cat photos & identify the cats
for f in imgs:

	# load the image & convert to greyscale
	image = cv2.imread(f, -1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# load the cat detector Haar cascade, then detect cat faces in the input image
	detector = cv2.CascadeClassifier(haarPath)
	rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))

	# loop over the cat faces and draw a rectangle surrounding each
	for (i, (x, y, w, h)) in enumerate(rects):
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

	# show the detected cat faces
	cv2.imshow("Cat Faces", image)
	cv2.waitKey(0)
