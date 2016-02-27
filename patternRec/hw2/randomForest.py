#Random forest implementation with bagged decision trees
import numpy as np
import helpers
from scipy import stats
from decisionTree import DecisionTree

class RandomForest(object):
  def __init__(self, numTrees):
    self.trees = [DecisionTree() for count in np.arange(numTrees)]


  def train(self, dataMatrix, classVectors):
    numToBag = int(helpers.HyperParams.FRAC_PER_BAG * dataMatrix.shape[0])

    augmentedDataMatrix = dataMatrix
    augmentedDataMatrix[classVectors.name] = classVectors

    for tree in self.trees:
      baggedData = helpers.bag(augmentedDataMatrix, numToBag)
      tree.train(baggedData.drop(classVectors.name, axis=1),
                 baggedData[classVectors.name])


  def predict(self, dataMatrix):
    predictions = np.empty((dataMatrix.shape[0]))
    for tree in self.trees:
      predictions = np.stack(tree.predict(dataMatrix))
    if len(predictions.shape)==1:
      return predictions
    else:
      return stats.mode(predictions, axis=0)[0]

