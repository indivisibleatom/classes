import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import lmdb
import caffe
from sklearn.metrics import confusion_matrix

def getKeyValuePairsFromLog(fileName, regex):
    file = open(fileName, 'r')
    keys = []
    values = []

    linesPrev = ["","",""]
    for line in file:
        linesPrev[0] = linesPrev[1]
        linesPrev[1] = linesPrev[2]
        linesPrev[2] = str.strip(line)
        lineCombined = linesPrev[0] + " " + linesPrev[1] + " " + linesPrev[2]
        match = regex.search(lineCombined)
        if match and match.group(1) not in keys:
           keys.append(match.group(1))
           values.append(match.group(2))
    return keys, values

def getTrainingLossFromTrainingFile(fileName):
    regex = re.compile("Iteration ([0-9]*).*Train net.*loss = ([0-9\.]*)" )
    return getKeyValuePairsFromLog(fileName, regex)

def getTestingLossFromTrainingFile(fileName):
    file = open(fileName, 'r')
    regex = re.compile("Iteration ([0-9]*).*Test net.*loss = ([0-9\.]*)")
    return getKeyValuePairsFromLog(fileName, regex)

def getAccuracyFromTrainingFile(fileName):
    file = open(fileName, 'r')
    regex = re.compile("Iteration ([0-9]*).*Test net.*accuracy = ([0-9\.]*)")
    return getKeyValuePairsFromLog(fileName, regex)

#Inspiration: https://www.snip2code.com/Snippet/559026/Caffe-script-to-compute-accuracy-and-con
#Read lmdb that is written by convert_imageset (which essentially serializes
#the data to the Datum type defined in caffe.proto)
def lmdb_reader(lmdb_env):
    lmdb_transaction = lmdb_env.begin()
    lmdb_cursor = lmdb_transaction.cursor()

    for key, value in lmdb_cursor:
        data = caffe.proto.caffe_pb2.Datum()
        data.ParseFromString(value)
        label = int(data.label)
        image = caffe.io.datum_to_array(data).astype(np.uint8)
        yield(key, image, label)

#Crop data with last two axes interpreted as dimension to desired shape
#Current uses center-crop
#TODO msati3: Handle differing dimensions
def crop(image, dataShape):
    imageShape = image.shape
    currentSize = np.array(imageShape[-2:])
    cropSize = np.array(dataShape[-2:])
    center = currentSize/2
    centeredWindow = cropSize / 2
    amountToRemove = center - centeredWindow
    if currentSize[0] <= cropSize[0] or currentSize[1] <= cropSize[1]:
        print "Request cropping of smaller image or equal to desired crop size"
    return (image[:,amountToRemove[0]:-amountToRemove[0]+1,
                   amountToRemove[1]:-amountToRemove[1]+1])


#Inspired from caffe examples
def showFilterGrid(filters, fGrayscale):
    tiledImage = (filters - filters.min()) / (filters.max() - filters.min())
    numGrids = int(np.ceil(np.sqrt(filters.shape[0])))
    padding = (((0, numGrids ** 2 - filters.shape[0]),
               (0, 1), (0, 1)) + ((0, 0),) * (filters.ndim - 3))
    tiledImage = np.pad(filters, padding, mode='constant', constant_values=1)

    tiledImage = tiledImage.reshape((numGrids, numGrids) +
                 tiledImage.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4,
                                                                 tiledImage.ndim + 1)))
    tiledImage = tiledImage.reshape((numGrids * tiledImage.shape[1], numGrids *
                 tiledImage.shape[3]) + tiledImage.shape[4:])
    if fGrayscale:
        plt.imshow(tiledImage[:,:,0], cmap='gray')
    else:
        plt.imshow(tiledImage)
    plt.axis('off')

def getConfusionMatrix(groundTruth, predictions):
    return confusion_matrix(groundTruth, predictions)

#Copied from scikit learn examples =>
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def showConfusionMatrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

