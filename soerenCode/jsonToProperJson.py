import pandas as pd
import numpy as np
import os,json,pickle
from scipy.signal import butter, lfilter



def butter_lowpass(cutoff, fs, order=4):
    fny = fs/2
    normal_cutoff = cutoff / fny
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def windowing(data,fs,windowSize):

    b,a = butter_lowpass(cutoff = 4 ,fs=fs)
    
    data = np.array(data)
    data = lfilter(b,a,data,axis=0)
    newTrials = []
    nSamples = len(data)
    start = 0
    end = windowSize

    while end < nSamples:
        newTrials.append(data[start:end,:])
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
    

