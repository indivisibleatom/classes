import numpy as np
import helpers

class HMMModel(object):
    def __init__(self):
        self.observationSymbols = ('r','g','b','y')
        self.__mappings = { (0,0): 0, (-1,0): 1, (0,-1): 2, (1,0): 3, (0,1): 4 }
        self.__inv = {0: (0,0), 1: (-1,0), 2: (0,-1), 3: (1,0), 4: (0,1)}

    def fit(self, X):
        countsList = [self.getHMMCounts(walk) for walk in X]
        sc, oc, tc = tuple(sum(x) for x in zip(*countsList))
        self.startP = helpers.divZero(sc, np.sum(sc))
        self.observationP = np.transpose(helpers.divZero(
                            oc, np.sum(oc, axis=0)),(1,2,0))
        # Note transitionP is indexable by neighbor,x,y
        self.transitionP = helpers.divZero(tc, np.sum(tc, axis=0))

    #Given observation and next observation (which can be None), gets
    #Von-Neuman neighborhoood index of next as index of [center left
    #top right bottom]
    def __getNeighbor(self, o, no):
        i1,j1 = o[0],o[1]
        i2,j2 = no[0],no[1]
        dif = (i2-i1, j2-j1)
        return self.__mappings[dif]

    def __getNeighborTuple(self, tup, no):
        neighbor = self.__inv[no]
        return (tup[0] + neighbor[0], tup[1] + neighbor[1])

    def getHMMCounts(self, walk):
        startCount = np.zeros((4,4))
        startCount[walk[0][0],walk[0][1]] = 1

        observationCounts = np.zeros((len(self.observationSymbols),4,4))
        transitionCounts = np.zeros((5,4,4))
        for o,no in zip(walk, walk[1:]+[None]):
            observationCounts[self.observationSymbols.index(o[2]),
                              o[0], o[1]] += 1
            if no is not None:
                transitionCounts[self.__getNeighbor(o, no),o[0],o[1]] += 1
        return (startCount,observationCounts,transitionCounts)

    #Implements the Viterbi dynamic programming algorithm for finding most
    #probable state transitions for given set of algorithms
    def scoreSingle(self, walk):
        transitionBackTrack = np.zeros((len(walk),4,4),int)
        optStateCurP = np.zeros((4,4))
        optStateCurP = (self.startP *
                 self.observationP[:,:,self.observationSymbols.index(walk[0][2])])
        for idx,o in enumerate(walk[1:]):
            optStateAllPTemp = (optStateCurP * self.transitionP)
            optStateNewP = np.zeros((4,4))
            #TODO msati3: Perhaps full transition map would be better as it
            #would allow for parallelization.
            for index,value in np.ndenumerate(optStateAllPTemp):
                n,x,y = index
                nt = self.__getNeighborTuple((x,y),n)
                if not (nt[0] > 3 or nt[1] > 3 or nt[0] < 0 or nt[1] < 0):
                    op = (self.observationP[nt[0], nt[1],
                                           self.observationSymbols.index(o[2])])
                    if (optStateAllPTemp[index] * op >
                            optStateNewP[nt[0],nt[1]]):
                       optStateNewP[nt[0],nt[1]] = optStateAllPTemp[index] * op
                       transitionBackTrack[(idx+1,nt[0],nt[1])] = (
                       np.ravel_multi_index((x,y),(4,4)))
            optStateCurP = optStateNewP
        optP = np.amax(optStateCurP)
        optT = np.argmax(optStateCurP)
        optT = np.unravel_index(optT, (4,4))
        optSeq = []
        for idx,val in enumerate(walk):
            optSeq.append(optT)
            transition = (transitionBackTrack[(199-idx,) + optT])
            optT = np.unravel_index(transition, (4,4))
        score = [1 if predTup == (obs[0],obs[1]) else 0 for predTup,obs in
                 zip(optSeq[::-1], walk)]
        list1 = [(predTup, (obs[0], obs[1])) for
                 predTup, obs in zip(optSeq[::-1], walk)]
        return(sum(score)/len(score))

    def score(self, walks):
        return(sum(self.scoreSingle(walk) for walk in walks)/len(walks))


