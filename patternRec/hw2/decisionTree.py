#Wrapper over decision tree
#TODO msati3: Change impl to self time permitting
from sklearn import tree

class DecisionTree(object):
    def __init__(self, maxDepth):
        self.tree = tree.DecisionTreeClassifier(max_depth=maxDepth)

    def train(self, dataMatrix, classVector):
        self.tree.fit(dataMatrix, classVector)

    def predict(self, dataMatrix):
        return self.tree.predict(dataMatrix)

