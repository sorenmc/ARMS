import os,sys
import pandas as pd
import numpy as np
import json,time
import tensorflow as tf
import filterSlidingWindow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Dropout,BatchNormalization,Conv2D,Conv1D,Flatten,LSTM,MaxPool1D,TimeDistributed
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix

class dataModel:
    def __init__(self):
        """
        loads, preprocess and splits data on start up. 
        """
        #initialize parameters

        self.xTrain = None
        self.yTrain = None
        self.xValidation = None
        self.yValidation = None
        self.xTest = None
        self.yTest = None
        self.nClasses = None
        self.models = None
        self.model = None
        

        self.loadPreprocessData()

        
    def loadPreprocessData(self):
        """
        Loads data from the file data.json. 
        Each trial is then filtered and windowed according to below parameters
        Afterwards the data is split stratified wise into training, validation and test sets.
        """
        fs = 32
        windowSize = fs*4
        slide = fs*1
        cutoff = 5
        order = 4
        labeledData = filterSlidingWindow.loadFileApplyfilterAndSlidingWindow(windowSize,slide,cutoff,order)
        self.stratifyData(labeledData)
    
    def stratifyData(self,data):
        """
        Given data is split stratified wise into 70% training, 15% validation and 15% test sets.
        """
        xTrain,xValidation,yTrain,yValidation =  train_test_split(data["trials"],data['labels'],test_size = 0.3,random_state=42,stratify=data['labels'])
        xValidation,xTest,yValidation,yTest =  train_test_split(xValidation,yValidation,test_size = 0.5,random_state=42,stratify=yValidation)

        encoder = LabelBinarizer()
        self.xTrain = np.array(xTrain).reshape((len(xTrain),128,3))
        self.xValidation = np.array(xValidation).reshape((len(xValidation),128,3))
        self.xTest = np.array(xTest).reshape((len(xTest),128,3))
        self.yTrainDecoded = yTrain
        self.yTrain = encoder.fit_transform(yTrain)
        self.yValidation = encoder.fit_transform(yValidation)
        self.yTest = encoder.fit_transform(yTest)
        
        #after transforming to one hot label vectors, the length is equal to the amount of classes
        self.nClasses = len(self.yTest[0]) 

    def buildNets1DLSTM(self):
        """
        According to below lists, this function will build and compile several versions
        of a 1d convolutional neural network followed by a LSTM
        """

        models = []

        denseLayers = [0]
        CNNLayers = [3]
        filters = [128]
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
                    #model.add(Flatten())
                    model.add(LSTM(100))

                    for _ in range(dense):
                        model.add(Dense(filt))
                        model.add(Activation("elu"))
                    
                    model.add(Dense(self.nClasses, activation = "softmax" ))
                    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
                    #model.summary()
                    keyVals = {"model":model,
                                "name":nameOfModel}
                    models.append(keyVals)
        
        self.models = models
    
    def crossTrainEval(self,name,epochs,batchSize):
        """
        This method will run 10x cross fold validation on the compiled model stored in 
        self.model
        """

        
        print("training network {}".format(name))
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        cross = skf.split(self.xTrain,self.yTrainDecoded)
        trainAccuracies = []
        evalAccuracies = []
        for train,test in cross:
            model= keras.models.clone_model(self.model)
            model.build((None, 10)) # replace 10 with number of variables in input layer
            model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
            model.set_weights(self.model.get_weights())
            model.fit(self.xTrain[train],self.yTrain[train],epochs = epochs, batch_size = batchSize, 
            verbose = 0, validation_data = (self.xTrain[test],self.yTrain[test]))
            trainAccuracies.append(model.evaluate(self.xTrain[train],self.yTrain[train])[1])
            evalAccuracies.append(model.evaluate(self.xTrain[test],self.yTrain[test])[1])
        accuracies = (np.mean(trainAccuracies),np.mean(evalAccuracies))
        print("crossval accuracies are {}".format(accuracies))
        return accuracies

    def trainNetwork(self,epochs,batchSize):
        """
        This method will train the current model stored in 
        self.model with validation data
        """
        print("training network")
        self.model.fit(self.xTrain,self.yTrain,epochs = epochs, batch_size = batchSize, 
        verbose = 1, validation_data = (self.xValidation,self.yValidation))
    
    def validateNetwork(self):
        """
        This method will get the validation accuracy of the trained model stored in 
        self.model
        """

        accuracy = self.model.evaluate(self.xValidation,self.yValidation)
        print("The accurcy on the Eval Data was: " + str(accuracy))
        return accuracy

    
    def testNetwork(self):
        """
        This method will get the unseen test accuracy of the trained model stored in 
        self.model
        """
        accuracy = self.model.evaluate(self.xTest,self.yTest)
        print("The accurcy on the test Data was: " + str(accuracy))


    
dataModel1 = dataModel()
dataModel1.buildNets1DLSTM()

for keyVals in dataModel1.models:
    dataModel1.model = keyVals["model"]
    name = keyVals["name"]
    print("10x crossfold validating {}".format(name))
    dataModel1.crossTrainEval(name,epochs=200,batchSize=100)
    print("training the {}".format(name))
    dataModel1.trainNetwork(epochs=200,batchSize =100)
    valAcc = dataModel1.validateNetwork()
    if valAcc >= 0.9:
        break
    
dataModel1.testNetwork()






    


                





        


        

