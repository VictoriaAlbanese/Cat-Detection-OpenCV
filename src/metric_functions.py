################################################################
#
# Filename: metric_functions.py
#
# Purpose: Contains different functions which make cool metrics
# these can oftentimes be big and unwieldy so we packed them in
# a separate file, all tidy!
#
################################################################

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import numpy as np

################################################################

# PLOT CONFUSION MATRIX FUNCTION
# this function plots a beautiful confusion matrix
def plot_confusion_matrix(y_actual, y_predict) :
    cm = confusion_matrix(y_actual, y_predict)
    print('Confusion Matrix:', cm)
    classes = [0, 1]
    class_labels = ["cat", "dog"]
    title = 'Confusion Matrix'
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_labels, yticklabels=class_labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

# PLOT ROC CURVE FUNCTION
# this function plots a beautiful roc curve
def plot_roc_curve(y_actual, y_predict) :
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(y_predict)) :
        fpr[i], tpr[i], _ = roc_curve(y_actual, y_predict)
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_actual, y_predict)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    plt.plot(fpr[2], tpr[2], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()