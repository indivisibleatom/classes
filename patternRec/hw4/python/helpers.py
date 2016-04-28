import sys
import re
import numpy as np
import matplotlib.pyplot as plt

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

#Inspired from online example code
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
