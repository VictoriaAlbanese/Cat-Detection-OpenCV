# import the necessary packages
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

import nn_functions as nn

from feature_extractors import baseline_extractor
from feature_extractors import intensity_extractor
from feature_extractors import edge_extractor
from feature_extractors import DOG_extractor
from feature_extractors import HOG_extractor

EXTRACTOR = intensity_extractor
CLASSES = ["cat", "dog"]

# load the network
print("[INFO] loading network architecture and weights...")
model_path = nn.get_full_path("\\output\\trained_network_{}.hdf5".format(EXTRACTOR.__name__))
model = load_model(model_path)

# load the images
data_path = "\\dataset\\test_data\\"
print("[INFO] loading images in {}".format(nn.get_full_path(data_path)))
data, labels = nn.extract_image_data(data_path, EXTRACTOR, debug=True)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(X, y,	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

# extract the predicted and actual labels
y_predict = [model.predict(X)[index] for (index, value) in enumerate(y)]
y_actual = [0 if value[0] == 1.0 else 1 for (index, value) in enumerate(y)]

'''
# draw the class and probability on the test image and display it to our screen
for (i, label) in enumerate(labels) :
	label = "{}: {:.2f}%".format(CLASSES[prediction], probs[prediction] * 100)
	cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
'''

'''
# confusion matrix stuff
print(confusion_matrix(actual_labels, predictions))

# print classification report 
# recall of positive class is sensitivity
# recall of negative class is specificity
print(classification_report(actual_labels, predictions))

# roc curve stuff
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(predictions)) :
    fpr[i], tpr[i], _ = roc_curve(actual_labels, predictions)
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(actual_labels, predictions)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''