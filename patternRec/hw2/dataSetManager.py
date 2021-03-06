#Manage different dataset loading, sampling etc
import pandas
import numpy as np

class WineDatasetManager(object):
    path = "wine/wine.data"
    headers = ['Lab','Alc','Ma','Ash','Alk','Mag','Phe','Fla','NoFla','Pro',
               'Col','Hue','OD','Pr']
    labelColumn = headers[0]

    def __init__(self):
        self.trainingData = ()
        self.testingData = ()

    @staticmethod
    def loadData():
        readData = pandas.read_csv(WineDatasetManager.path, names =
                                   WineDatasetManager.headers)
        readData.dropna(how='all', inplace='True')
        WineDatasetManager.data = readData

    def populateTrainTest(self, percentTraining):
        self.trainingData = WineDatasetManager.data.sample(frac=percentTraining)
        self.testingData = WineDatasetManager.data.drop(self.trainingData.index)

class MNISTDatasetManager(object):
    pathTrain = "MNIST/train.csv"
    pathTest = "MNIST/test.csv"
    labelColumn = 784

    def __init__(self):
        self.trainingData = ()
        self.testingData = ()

    @staticmethod
    def loadData():
        MNISTDatasetManager.dataTrain = pandas.read_csv(
                          MNISTDatasetManager.pathTrain, header=None)
        MNISTDatasetManager.dataTest = pandas.read_csv(
                          MNISTDatasetManager.pathTest, header=None)

    def populateTrainTest(self):
        self.trainingData = MNISTDatasetManager.dataTrain.transpose()
        self.testingData = MNISTDatasetManager.dataTest.transpose()

def loadDataSets():
    WineDatasetManager.loadData()
    MNISTDatasetManager.loadData()

#Perform these on import
loadDataSets()
