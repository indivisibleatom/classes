#Random forest implementation with bagged decision trees
import numpy as np
import numpy.ma as ma
import pandas
import helpers
from scipy.stats import mstats
from decisionTree import DecisionTree

class RandomForest(object):
  def __init__(self, numTrees, treeMaxDepth=None):
    self.trees = [DecisionTree(treeMaxDepth) for count in np.arange(numTrees)]
    self.baggedIndicesBoolean = [] #One for each tree if bagged
    self.boostedBeta = [] #One for each tree if boosted

  def bagAndTrain(self, tree, dataMatrix, classVectors):
    numToBag = int(helpers.HyperParams.FRAC_PER_BAG * dataMatrix.shape[0])
    baggedIndices = helpers.bag(np.arange(dataMatrix.shape[0]), numToBag)
    tree.train(dataMatrix.iloc[baggedIndices], classVectors.iloc[baggedIndices])
    baggedIndicesBool = np.zeros(dataMatrix.shape[0], dtype=bool)
    baggedIndicesBool[baggedIndices] = True
    return baggedIndicesBool

  def trainBagged(self, dataMatrix, classVectors):
    self.baggedIndicesBoolean = [self.bagAndTrain(tree, dataMatrix,
                        classVectors) for tree in self.trees]
    self.trainingData = dataMatrix

  # Returns dataIndex X mislabelIndex X correctlabelIndex
  def getMislabelSet(self, dataIndices, classVectors):
    classes = pandas.unique(classVectors)
    mislabelArray = np.repeat(np.arange(classes.size), dataIndices.size)
    correctLabelRepeated = np.tile(np.where(classes == classVectors.values),
                                   classes.size)
    mislabelArray = mislabelArray[mislabelArray != correctLabelRepeated]
    mislabelSet = np.transpose([np.tile(dataIndices, classes.size - 1),
                               mislabelArray,
                               np.tile(np.where(classes == classVectors.values),
                               classes.size - 1)])
    print(np.where(classes == classVectors.values[10000]))
    return mislabelSet

  def calculatePseudoLoss(self, tree, dataMatrix, mislabels, distribution):
    probs = tree.predictProbabilities(dataMatrix.iloc[mislabels[:,0]])
    probsIncorrect = probs[:,mislabels[:,1]]
    probsCorrect = probs[:,mislabels[:,2]]
    loss = 0.5 * sum(distribution * (1 - probs[:,2] + probs[:,1]))
    return loss

  def updateDistribution(self, dataMatrix, mislabels, distribution, beta):
    probs = tree.predictProbabilities(dataMatrix.iloc[mislabels[:,0]])
    power = (0.5 * sum(distribution * (1 + probs[:,2] - probs[:,1])))
    distribution = distribution * (beta ^ power)

  def trainBoosted(self, dataMatrix, classVectors):
    mislabels = self.getMislabelSet(np.arange(
                 dataMatrix.shape[0]), classVectors)
    distribution = np.ones(mislabels.shape[0])
    distribution = helpers.toProbDistribution(distribution)
    numToBag = int(helpers.HyperParams.FRAC_PER_BAG * mislabels.shape[0])
    mislabelIndices = helpers.bag(np.arange(mislabels.shape[0]),
                                  numToBag, distribution)
    for tree in self.trees:
        tree.train(dataMatrix.iloc[mislabels[mislabelIndices][:,0]],
                   classVectors.iloc[mislabels[mislabelIndices][:,0]])
        loss = self.calculatePseudoLoss(tree, dataMatrix,
                                        mislabels, distribution)
        beta = loss/(1-loss)
        self.boostedBeta.append(beta)
        distribution = self.updateDistribution(dataMatrix, mislabels,
                                               distribution, beta)

  def predictBoosted(self, dataMatrix):
    predictions = sum([-log(beta) * tree.predictProbabilities(dataMatrix)
                     for tree,beta in zip(self.trees,self.boostedBeta)])

  # Make out of bag predictions using cached training data
  def predictOutOfBagTraining(self):
    predictions = [ma.masked_array(tree.predict(
                   self.trainingData), mask=oobIndices)
                   for tree,oobIndices in
                   zip(self.trees, self.baggedIndicesBoolean)]
    return self.calcMode(predictions)

  def predict(self, dataMatrix):
    predictions = [ma.masked_array(tree.predict(dataMatrix), mask=False)
                   for tree in self.trees]
    return self.calcMode(predictions)

  def calcMode(self, predictions):
    predictions = np.ma.vstack(predictions)
    return mstats.mode(predictions, axis=0)

