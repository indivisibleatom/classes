#Manage different dataset loading, sampling etc
import pandas
import numpy as np
from sklearn.cross_validation import train_test_split

class WineDatasetManager(object):
    path = "wine/wine.data"
    headers = ['Lab','Alc','Ma','Ash','Alk','Mag','Phe','Fla','NoFla','Pro',
               'Col','Hue','OD','Pr']
    labelColumn = headers[0]

    def __init__(self):
        self.trainingData = ()
        self.testingData = ()
        self.loadData()

    def loadData(self):
        readData = pandas.read_csv(WineDatasetManager.path, names =
                                   WineDatasetManager.headers)
        readData.dropna(how='all', inplace='True')
        self.data = readData

    def populateTrainTest(self, percentTraining):
        self.trainingData, self.testingData = train_test_split(self.data)

class MNISTDatasetManager(object):
    pathTrain = "MNIST/train.csv"
    pathTest = "MNIST/test.csv"
    labelColumn = 784

    def __init__(self):
        self.trainingData = ()
        self.testingData = ()
        self.loadData()

    def loadData(self):
        self.dataTrain = pandas.read_csv(
                          MNISTDatasetManager.pathTrain, header=None)
        self.dataTest = pandas.read_csv(
                          MNISTDatasetManager.pathTest, header=None)

    def populateTrainTest(self):
        self.trainingData = self.dataTrain.transpose()
        self.testingData = self.dataTest.transpose()

class RobotDatasetManager(object):
    pathTrain = "Robot/robot_train.data"
    pathTest = "Robot/robot_test.data"

    def __init__(self):
        self.trainingData = self.readData(RobotDatasetManager.pathTrain)
        self.testingData = self.readData(RobotDatasetManager.pathTest)

    def readData(self, path):
        f = open(path, 'r')
        walks = []
        currentList = []
        for line in f:
            if line == "." or line ==".\n":
                walks.append(currentList)
                currentList = []
            else:
                i,j,c = line[0],line[2],line[4]
                currentList.append((int(i)-1,int(j)-1,c))
        #Patch up files that don't have terminal dot
        if line != "." and line !=".\n":
            walks.append(currentList)
        return walks

