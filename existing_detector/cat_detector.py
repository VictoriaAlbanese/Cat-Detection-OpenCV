# import the necessary packages
import cv2
import os
from glob import glob

# construct the argument parse and parse the arguments
dataPath = os.getcwd() + "\\dataset\\"
haarPath = os.getcwd() + "\\existing_detector\\haarcascade_frontalcatface.xml"
imgs = [f for f in glob(dataPath + 'training_data\\*') if '.jpg' in f]

# iterate through all the cat photos & identify the cats
for (i, f) in enumerate(imgs):

	# load the image & convert to greyscale
	image = cv2.imread(f, -1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# load the cat detector Haar cascade, then detect cat faces in the input image
	detector = cv2.CascadeClassifier(haarPath)
	rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))

	for (j, (x, y, w, h)) in enumerate(rects) :
		crop_img = image[y:y+h, x:x+w]
		img_name = os.getcwd() + '\\dataset\\cropped_imgs\\{}'.format(f[f.rfind("\\") + 1:])
		cv2.imwrite(img_name, crop_img)

	if i > 0 and i % 100 == 0 :
		print("[INFO] processed {}/{}".format(i, len(imgs)))
	


	# loop over the cat faces and draw a rectangle surrounding each
	#for (i, (x, y, w, h)) in enumerate(rects):
	#	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	#	cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

	# show the detected cat faces
	#cv2.imshow("Cat Faces", image)
	#cv2.waitKey(0)
