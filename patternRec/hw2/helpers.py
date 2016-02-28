#Helper functions
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix

class HyperParams(object):
  FRAC_PER_BAG = 1/3

#Given dataframe, bag it by selecting n with replacement
def bag(data, numItems):
  assert(type(data) == pandas.DataFrame)
  baggedData = data.sample()
  for numItem in np.arange(numItems-1):
    baggedData = baggedData.append(data.sample())
  assert(baggedData.shape[0] == numItems), """Number of items bagged (%d) 
  is not same as the number desired (%d):"""  %(baggedData.shape[0], numItems)
  return baggedData

#Given two vectors, compute differing elements fraction. The first argument
#is the ground truth, and the second the predictions
def computeClassificationError(truth, prediction):
  return (np.sum([truth == prediction]) / truth.shape)[0]

def confusionMatrix(truth, prediction):
  return confusion_matrix(truth, prediction)
