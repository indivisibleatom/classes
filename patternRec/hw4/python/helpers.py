import sys
import re
import numpy as np
import matplotlib.pyplot as plt

def getLossFromTrainingFile(fileName):
    file = open(fileName, 'r')
    iterations = []
    losses = []
    regex = re.compile("Iteration ([0-9]*).*?loss = ([0-9\.]*)")
    for line in file:
        line = str.strip(line)
        match = regex.search(line)
        if match:
           iterations.append(match.group(1))
           losses.append(match.group(2))
    return iterations, losses

def getAccuracyFromTrainingFile(fileName):
    file = open(fileName, 'r')
    iterations = []
    accuracies = []
    regex1 = re.compile("Iteration ([0-9]*).*Testing net")
    regex2 = re.compile("Test net output.*accuracy = ([0-9\.]*)")
    for line in file:
        line = str.strip(line)
        match1 = regex1.search(line)
        if match1:
           iterations.append(match1.group(1))
        match2 = regex2.search(line)
        if match2:
           accuracies.append(match2.group(1))
    return iterations, accuracies

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
