#Implements KNN Classifier.
import numpy as np
import bottleneck as bn
from scipy import stats
from metric_learn import LMNN
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

# KNN classifier. Implements fit to learn a distance metric and score to score
# a test set.
class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1):
        self.k = k
        self.distanceEstimator = LMNN(k=k)

    def fit(self, X, y):
        #TODO msati3: Ideally, LMNN should expose fit_transform.
        self.distanceEstimator.fit(X, y)
        self.modelData = self.distanceEstimator.transform(X)
        self.modelLabels = y
        return self

    def transform(self, X):
        return self.distanceEstimator.transform(X)

    def predict(self, D):
        X = self.transform(D) #Pretransform so that euclidean metric suffices
        distances = distance.cdist(X, self.modelData,'sqeuclidean')
        topKIndexes = bn.argpartsort(distances, self.k)[:,:self.k]
        predictions = self.modelLabels[topKIndexes]
        return stats.mode(predictions, axis=1)[0]

    def score(self, X, y, fNormalize=True):
        return accuracy_score(self.predict(X), y, fNormalize)

