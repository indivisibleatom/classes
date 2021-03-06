#Helper functions
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import LeaveOneOut, KFold
import matplotlib.pyplot as plt

class HyperParams(object):
  FRAC_PER_BAG = 1/3

#Divide that replaces 0/0 with 0
def divZero(a,b):
    with np.errstate(divide='ignore'):
        div = a/b
    return np.nan_to_num(div)

#Given two vectors, compute differing elements fraction. The first argument
#is the ground truth, and the second the predictions
def computeClassificationError(truth, prediction):
  return (np.sum([truth == prediction]) / truth.shape)[0]

def confusionMatrix(truth, prediction):
  return confusion_matrix(truth, prediction)

def toProbDistribution(distribution):
  return distribution/sum(distribution)

#Input is a dataframe. Samples with accept probability. Keeps distribution of
#class labels
def sample(data, labelColumn, acceptProbability, fReplace):
  classes = np.unique(data[labelColumn])
  classDatas = [data[data[labelColumn] == labelClass] for labelClass in classes]
  sampled = [data.sample(frac=acceptProbability, replace=fReplace) for data in
             classDatas]
  return pd.concat(sampled)

#Get appropriate sklear.cv iterator, handling sentinel 0 for LeaveOneOut
def getCrossValidationIterator(n, n_folds):
    return LeaveOneOut(n) if n_folds == 0 else KFold(n, n_folds)

#Given a 3D array representing a gridded state table, with a[x,y,:] 
#corresponding to probabilities at state x,y, plots them in the figure using a
# stacked bar chart at x,y
def plotLatticeStackedProbabilities(array, colors):
    fig = plt.figure()
    increments = np.asarray((1,1))/np.asarray(array.shape[:2])
    for index in np.ndindex(array.shape[:2]):
        rect = tuple(np.asarray(index)*increments) + tuple(increments)
        ax = fig.add_axes(rect)
        cumsum = 0
        for indexProbs in np.arange(array.shape[-1]):
            probability = array[index+(indexProbs,)]
            ax.bar(0, probability, 1, color=colors[indexProbs],
                   bottom = cumsum)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            cumsum += probability
        #Fill up the slot with filler color if probs don't sum to 1
        ax.bar(0, 1-cumsum, 1, color=colors[-1], bottom = cumsum)

