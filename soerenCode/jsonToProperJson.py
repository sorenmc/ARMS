import pandas as pd
import numpy as np
import os,json,pickle

def windowing(f,fs,windowSize):
    f = np.array(f)
    newTrials = []
    nSamples = len(f)
    start = 0
    end = windowSize

    while end < nSamples:
        newTrials.append(f[start:end,:])
        start += windowSize
        end += windowSize
    
    return newTrials


with open("data.json","r") as fin:
    jsonData = json.load(fin)

fs = 32


#4 second windows
windowSize = fs*4 
newJson = {}
for key in jsonData.keys():
    category = jsonData[key]
    newCategory = []
    for f in category:
        newCategory.extend(windowing(f,fs,windowSize))
    if key[-5:] == "MODEL":
        print("not gonna use " + str(key))
    else:    
        newJson[key] = newCategory

jsonDataFrame ={}
jsonDataFrame["labels"] = []
jsonDataFrame["trials"] = []
for key, data in newJson.items():
    for array in data:
        jsonDataFrame["labels"].append(key)
        jsonDataFrame["trials"].append(array.tolist())

with open("labeledAccelerometer.json",'w') as fout:
    json.dump(jsonDataFrame,fout)

#df.to_csv("labeledAccelerometer.csv", sep='\t',encoding='utf-8')
a=1
    

