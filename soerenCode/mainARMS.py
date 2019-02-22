import os,sys
import pandas as pd
import numpy as np
import json,time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Dropout, BatchNormalization,Conv2D,Conv1D,Flatten,LSTM,MaxPool1D
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

class dataModel:
    def __init__(self,dataName):
        self.xTrain = 0
        self.yTrain = 0
        self.xEval = 0
        self.yEval = 0
        self.xTest = 0
        self.yTest = 0
        self.nClasses = 0

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
        self.xTrain = np.array(xTrain).reshape((len(xTrain),128,3))
        self.xEval = np.array(xEval).reshape((len(xEval),128,3))
        self.xTest = np.array(xTest).reshape((len(xTest),128,3))
        self.yTrain = encoder.fit_transform(yTrain)
        self.yEval = encoder.fit_transform(yEval)
        self.yTest = encoder.fit_transform(yTest)
        
        #after transforming to one hot label vectors, the length is equal to the amount of classes
        self.nClasses = len(self.yTest[0])
    
    def buildNet2D(self):
        model = Sequential()
        model.add(Conv2D(input_shape = (128,3,1,),kernel_size = (8,3), padding = "valid", strides = (2,1), filters = 3, activation = "relu"))
        model.add(BatchNormalization())
        #model.add(Dropout(0.5))
        model.add(Conv2D(kernel_size = (11,1), strides = (2,1), padding = "valid", filters = 1, activation = "elu"))
        model.add(BatchNormalization())
        model.add(Flatten())
        #model.add(Dropout(0.5))
        model.add(Dense(self.nClasses, activation = "softmax" ))
        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        self.model = model
        print(model.summary())

    
    def buildNets1D(self):

        models = []

        denseLayers = [0,1,2]
        CNNLayers = [1,2,3,4,5]
        filters = [2,4,8,16,32,64]
        for dense in denseLayers:
            for CNNLayer in CNNLayers:
                for filt in filters:
                    nameOfModel = "{}-conv-{}-filter-{}-dense-{}".format(CNNLayer,filt,dense,int(time.time()))
                    model = Sequential()
                    model.add(Conv1D(input_shape = (128,3,),kernel_size = (3), padding = "valid", filters = filt))
                    model.add(Activation("elu"))
                    model.add(BatchNormalization())
                    model.add(MaxPool1D(pool_size=2,padding="valid"))
                    
                    for _ in range(CNNLayer):
                        model.add(Conv1D(kernel_size = (2), padding = "valid", filters = filt))
                        model.add(Activation("elu"))
                        model.add(BatchNormalization())
                        model.add(MaxPool1D(pool_size=2,padding="valid"))
        
                    model.add(Flatten())

                    for _ in range(dense):
                        model.add(Dense(filt))
                        model.add(Activation("elu"))
                    
                    model.add(Dense(self.nClasses, activation = "softmax" ))
                    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

                    keyVals = {"model":model,
                                "name":nameOfModel}
                    models.append(keyVals)
        
        self.models = models
    
    def trainNetwork(self,epochs,batchSize):
        print("training network")
        self.model.fit(self.xTrain,self.yTrain,epochs = epochs, batch_size = batchSize, 
        verbose = 1, validation_data = (self.xEval,self.yEval))
    
    def validateNetwork(self):
        accuracy = self.model.evaluate(self.xEval,self.yEval)
        print("The accurcy on the Eval Data was: " + str(accuracy))

    
    def testNetwork(self):
        accuracy = self.model.evaluate(self.xTest,self.yTest)
        print("The accurcy on the test Data was: " + str(accuracy))


    
dataModel1 = dataModel("labeledAccelerometer.json")
#dataModel1.buildNet2D()
dataModel1.buildNets1D()

wFile = open("results.text","w")

for keyVals in dataModel1.models:
    dataModel1.model = keyVals["model"]
    name = keyVals["name"]
    
    wFile.write("\n")
    wFile.write(name)

    dataModel1.trainNetwork(epochs=100,batchSize =128)
    sys.stdout = wFile
    
    dataModel1.validateNetwork()
    sys.stdout = sys.__stdout__

wFile.close()





    


                





        


        

