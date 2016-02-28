#Random forest implementation with bagged decision trees
import numpy as np
import numpy.ma as ma
import helpers
from scipy.stats import mstats
from decisionTree import DecisionTree

class RandomForest(object):
  def __init__(self, numTrees, treeMaxDepth=None):
    self.trees = [DecisionTree(treeMaxDepth) for count in np.arange(numTrees)]
    self.outOfBagIndices = [] #One for each tree

  def trainBagged(self, dataMatrix, classVectors):
    numToBag = int(helpers.HyperParams.FRAC_PER_BAG * dataMatrix.shape[0])

    augmentedDataMatrix = dataMatrix.copy()
    print(augmentedDataMatrix.shape)
    augmentedDataMatrix[classVectors.name] = classVectors

    for tree in self.trees:
      baggedData = helpers.bag(augmentedDataMatrix, numToBag)
      tree.train(baggedData.drop(classVectors.name, axis=1),
                 baggedData[classVectors.name])
      self.outOfBagIndices.append(~dataMatrix.index.isin(baggedData.index))
    self.trainingData = dataMatrix

  # Make out of bag predictions using cached training data
  def predictOutOfBagTraining(self):
    predictions = [ma.masked_array(tree.predict(
                   self.trainingData), mask=~oobIndices)
                   for tree,oobIndices in
                   zip(self.trees, self.outOfBagIndices)]
    return self.calcMode(predictions)

  def predict(self, dataMatrix):
    predictions = [ma.masked_array(tree.predict(dataMatrix), mask=False)
                   for tree in self.trees]
    return self.calcMode(predictions)

  def calcMode(self, predictions):
    predictions = np.ma.vstack(predictions)
    return mstats.mode(predictions, axis=0)

