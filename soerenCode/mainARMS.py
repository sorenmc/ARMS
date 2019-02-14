import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Dropout, BatchNormalization, Conv2D, Flatten
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split
class dataModel:
    def __init__(self,dataName):
        self.xTrain = 0
        self.yTrain = 0
        self.xEval = 0
        self.yEval = 0
        self.xTest = 0
        self.yTest = 0

        self.model = 0

        self.loadData(dataName)

        
    def loadData(self,dataName):
        with open(dataName,'r') as fileIn:
            data = json.load(fileIn)
        self.stratifyData(data)
    
    def stratifyData(self,data):
        xTrain,xEval,yTrain,yEval =  train_test_split(data["trials"],data['labels'],test_size = 0.3,random_state=42,stratify=data['labels'])
        xEval,xTest,yEval,yTest =  train_test_split(xEval,yEval,test_size = 0.5,random_state=42,stratify=yEval)

        #encoder = LabelEncoder()
        encoder = LabelBinarizer()
        #transformedLabaelencoder.fit_transform(list(set(yTrain)))
        #transfomed_label = encoder.fit_transform(["dog", "cat", "bird"])
        self.xTrain = np.array(xTrain).reshape((len(xTrain),128,3,1))
        self.xEval = np.array(xEval).reshape((len(xEval),128,3,1))
        self.xTest = np.array(xTest).reshape((len(xTest),128,3,1))
        self.yTrain = encoder.fit_transform(yTrain)
        self.yEval = encoder.fit_transform(yEval)
        self.yTest = encoder.fit_transform(yTest)
    
    def buildNet(self):
        nClasses = 14
        model = Sequential()
        model.add(Conv2D(input_shape = (128,3,1,),kernel_size = (8,3), padding = "valid", strides = (2,1), filters = 3, activation = "relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Conv2D(kernel_size = (11,1), strides = (2,1), padding = "valid", filters = 1, activation = "relu"))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(nClasses, activation = "softmax" ))
        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        self.model = model
        print(model.summary())
    
    def trainNetwork(self,epochs,batchSize):
        print("training network")
        self.model.fit(self.xTrain,self.yTrain,epochs = epochs, batch_size = batchSize, 
        verbose = 1, validation_data = (self.xEval,self.yEval))
    
    def validateNetwork(self):
        accuracy = self.model.evaluate(self.xTest,self.yTest)
        print("The accurcy on the test Data was: " + str(accuracy))

    
dataModel1 = dataModel("labeledAccelerometer.json")
dataModel1.buildNet()
dataModel1.trainNetwork(epochs=100,batchSize =128)
dataModel1.validateNetwork()





    


                








        


        

