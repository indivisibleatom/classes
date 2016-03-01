#Random forest implementation with bagged decision trees
import numpy as np
import numpy.ma as ma
import pandas
import math
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
    mislabelMask = [classes != classVector[1] for classVector in
                    np.ndenumerate(classVectors.values)]
    mislabelMask = np.stack(mislabelMask)
    mislabelMask = mislabelMask.flatten();
    mislabelArray = np.tile(np.arange(classes.shape[0]), dataIndices.size);
    labelArray = mislabelArray[~mislabelMask]
    labelArray = np.repeat(labelArray, classes.size - 1)
    mislabelArray = mislabelArray[mislabelMask]
    #Debug
    #for index in np.arange(dataIndices.size):
    #    print(mislabelMask[3*index:3*index+3],
    #          mislabelArray[3*index:3*index+3],
    #          labelArray[3*index:3*index+3])
    mislabelSet = np.transpose([np.tile(dataIndices, classes.size - 1),
                               mislabelArray, labelArray])
    return mislabelSet

  def calculatePseudoLoss(self, tree, dataMatrix, mislabels, distribution):
    probs = tree.predictProbabilities(dataMatrix.iloc[mislabels[:,0]])
    probsIncorrect = np.choose(mislabels[:,1], probs.T)
    probsCorrect = np.choose(mislabels[:,2], probs.T)
    loss = 0.5 * sum(distribution * (1 - probsCorrect + probsIncorrect))
    return loss

  def updateDistribution(self, tree, dataMatrix, mislabels, distribution, beta):
    probs = tree.predictProbabilities(dataMatrix.iloc[mislabels[:,0]])
    probsIncorrect = np.choose(mislabels[:,1], probs.T)
    probsCorrect = np.choose(mislabels[:,2], probs.T)
    power = (0.5 * sum(distribution * (1 + probsCorrect - probsIncorrect)))
    distribution = distribution * (np.power(beta,power))
    distribution = helpers.toProbDistribution(distribution)
    return distribution

  def trainBoosted(self, dataMatrix, classVectors):
    mislabels = self.getMislabelSet(np.arange(
                 dataMatrix.shape[0]), classVectors)
    distribution = np.ones(mislabels.shape[0])
    distribution = helpers.toProbDistribution(distribution)
    numToBag = int(mislabels.shape[0])

    for tree in self.trees:
        mislabelIndices = helpers.bag(np.arange(mislabels.shape[0]),
                                  numToBag, distribution)
        tree.train(dataMatrix.iloc[mislabels[mislabelIndices][:,0]],
                   classVectors.iloc[mislabels[mislabelIndices][:,0]])
        loss = self.calculatePseudoLoss(tree, dataMatrix,
                                        mislabels, distribution)
        beta = loss/(1-loss)
        self.boostedBeta.append(beta)
        distribution = self.updateDistribution(tree, dataMatrix, mislabels,
                                               distribution, beta)

  def predictBoosted(self, dataMatrix):
    predictions = np.sum(np.stack([-math.log(beta) *
                  tree.predictProbabilities(dataMatrix)
                  for tree,beta in zip(self.trees,self.boostedBeta)]), axis=0)
    predictions = np.argmax(predictions, axis=1)
    return predictions

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

