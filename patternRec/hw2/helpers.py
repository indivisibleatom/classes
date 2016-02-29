#Helper functions
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix

class HyperParams(object):
  FRAC_PER_BAG = 1/3

#Given indices, bag them by selecting n with replacement
def bag(dataIndices, numItems, probability=None):
  baggedIndices = np.random.choice(dataIndices, size=numItems, p=probability)
  assert(baggedIndices.shape[0] == numItems), """Number of items bagged (%d) 
  is not same as the number desired (%d):"""  %(baggedIndices.shape[0], numItems)
  return baggedIndices

#Given two vectors, compute differing elements fraction. The first argument
#is the ground truth, and the second the predictions
def computeClassificationError(truth, prediction):
  return (np.sum([truth == prediction]) / truth.shape)[0]

def confusionMatrix(truth, prediction):
  return confusion_matrix(truth, prediction)

def toProbDistribution(distribution):
  return distribution/sum(distribution)
