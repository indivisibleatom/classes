#Manage different dataset loading, sampling etc
import pandas
import numpy as np

class WineDatasetManager(object):
    path = "wine/wine.data"
    headers = ['Lab','Alc','Ma','Ash','Alk','Mag','Phe','Fla','NoFla','Pro',
               'Col','Hue','OD','Pr']

    def __init__(self):
        self.trainingData = ()
        self.testingData = ()

    @staticmethod
    def loadData():
        readData = pandas.read_csv(WineDatasetManager.path, names =
                                   WineDatasetManager.headers)
        readData.dropna(how='all', inplace='True')
        WineDatasetManager.data = readData

    def splitTrainTest(self, percentTraining):
        self.trainingData = WineDatasetManager.data.sample(frac=percentTraining)
        self.testingData = WineDatasetManager.data.drop(self.trainingData.index)

def loadDataSets():
    WineDatasetManager.loadData()


#Perform these on import
loadDataSets()
