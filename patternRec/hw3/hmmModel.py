import numpy as np
import helpers

class HMMModel(object):
    def __init__(self):
        self.observationSymbols = ('r','g','b','y')

    def fit(self, X):
        countsList = [self.getHMMCounts(walk) for walk in X]
        sc, oc, tc = tuple(sum(x) for x in zip(*countsList))
        self.startP = helpers.divZero(sc, np.sum(sc))
        self.observationP = np.transpose(helpers.divZero(
                            oc, np.sum(oc, axis=0)),(1,2,0))

    def getHMMCounts(self, walk):
        startCount = np.zeros((4,4))
        startCount[walk[0][0],walk[0][1]] = 1

        observationCounts = np.zeros((len(self.observationSymbols),4,4))
        transitionCounts = np.zeros((4,4,5))
        for observation in walk:
            observationCounts[self.observationSymbols.index(observation[2]),
                              observation[0], observation[1]] += 1
            #transitionCounts[i,j,getNeighbor(] = transitionCounts[i][j] + 
        return (startCount,observationCounts,startCount)
