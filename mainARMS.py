import os
import pandas as pd
import numpy as np

class dataModel:
    def __init__(self,trainData1,trainData2):
        evalExtension = "_MODEL"
        self.trainData = [self.loadData(trainData1),self.loadData(trainData2)] 
        self.evalData = [self.loadData(trainData1+evalExtension),self.loadData(trainData2+evalExtension)] 
        self.trainLabels = np.hstack((np.ones(len(self.trainData[0]),dtype = int),np.zeros(len(self.trainData[1]),dtype = int)))
        self.evalLabels = np.hstack((np.ones(len(self.evalData[0]),dtype = int),np.zeros(len(self.evalData[1]),dtype = int)))
        a=1
    def loadData(self,dataName):
        base = os.path.dirname(__file__)
        dataPath = os.path.abspath(os.path.join(base, "..", "HMP_Dataset", dataName))
        dataDirectory = os.fsencode(dataPath)
        allData = []
        for fileName in os.listdir(dataPath):
            filePath = os.path.join(dataPath,fileName)
            allData.append(pd.read_csv(filePath,sep= " ").values)
        return allData
                

climbingStairs = dataModel("Climb_stairs","Drink_glass")


        


        

