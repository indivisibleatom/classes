#Helper functions
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import LeaveOneOut, KFold

class HyperParams(object):
  FRAC_PER_BAG = 1/3

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
