import os,json,time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Dropout, BatchNormalization,Conv2D,Conv1D,Flatten,LSTM,MaxPool1D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from tensorflow.keras.callbacks import TensorBoard
from copy import deepcopy


class dataModel:
    def __init__(self):
        self.xTrain = None
        self.yTrain = None
        self.xEval = None
        self.yEval = None
        self.xTest = None
        self.yTest = None
        self.nClasses = None
        self.model = None

        self.currentLoadData()
        

    def currentLoadData(self):
        with open("labeledAccelerometer.json") as data:
            jData = json.load(data)
        self.stratifyData(jData)
      
    def stratifyData(self,data):
        xTrain,xEval,yTrain,yEval =  train_test_split(data["trials"],data["labels"],test_size = 0.3,random_state=42,stratify=data["labels"])
        xEval,xTest,yEval,yTest =  train_test_split(xEval,yEval,test_size = 0.5,random_state=42,stratify=yEval)

        #encoder = LabelEncoder()
        encoder = LabelBinarizer()
        #transformedLabaelencoder.fit_transform(list(set(yTrain)))
        #transfomed_label = encoder.fit_transform(["dog", "cat", "bird"])
        self.xTrain = np.array(xTrain).reshape((len(xTrain),128,3))
        self.xEval = np.array(xEval).reshape((len(xEval),128,3))
        self.xTest = np.array(xTest).reshape((len(xTest),128,3))
        self.yTrainDecoded = yTrain
        self.yTrain = encoder.fit_transform(yTrain)
        self.yEval = encoder.fit_transform(yEval)
        self.yTest = encoder.fit_transform(yTest)
        
        #after transforming to one hot label vectors, the length is equal to the amount of classes
        self.nClasses = len(self.yTest[0])

    def buildNets1D(self):

        models = []

        denseLayers = [0]
        CNNLayers = [3]
        filters = [64]
        for dense in denseLayers:
            for CNNLayer in CNNLayers:
                for filt in filters:
                    nameOfModel = "{}-conv-{}-filter-{}-dense-{}".format(CNNLayer,filt,dense,int(time.time()))
                    model = Sequential()
                    model.add(Conv1D(input_shape = (128,3,),kernel_size = (3), padding = "valid", filters = filt))
                    model.add(Activation("elu"))
                   #model.add(Dropout(0.1))
                    model.add(BatchNormalization())
                    model.add(MaxPool1D(pool_size=2,padding="valid"))
                    
                    for _ in range(CNNLayer):
                        model.add(Conv1D(kernel_size = (2), padding = "valid", filters = filt))
                        model.add(Activation("elu"))
                        #model.add(Dropout(0.1))
                        model.add(BatchNormalization())
                        model.add(MaxPool1D(pool_size=2,padding="valid"))
        
                    model.add(Flatten())

                    for _ in range(dense):
                        model.add(Dense(filt))
                        model.add(Activation("elu"))
                        model.add(Dropout(0.5))
                        model.add(BatchNormalization())
                    
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

    def crossTrainEval(self,name,epochs,batchSize):
        
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
        
        

    def validateNetwork(self):
        accuracy = self.model.evaluate(self.xEval,self.yEval)
        print("The accurcy on the Eval Data was: " + str(accuracy))

    
    def testNetwork(self):
        accuracy = self.model.evaluate(self.xTest,self.yTest)
        print("The accurcy on the test Data was: " + str(accuracy))

    
stats = pd
dM = dataModel()
dM.buildNets1D()
scores= {'name': [],
         'train': [],
         'eval': []
        }

for keyVals in dM.models:
    dM.model = keyVals["model"]
    name = keyVals["name"]
    trainCross,evalCross = dM.crossTrainEval(name,200,100)
    scores['name'].append(name) 
    scores['train'].append(trainCross)
    scores['eval'].append(evalCross)

a = 1
df = pd.DataFrame.from_dict(scores)
df.to_csv("results.csv")
#sdf = spark.createDataFrame(df)
#sdf.write.saveAsTable("results")    