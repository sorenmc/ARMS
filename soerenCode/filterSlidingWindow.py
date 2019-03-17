import pandas as pd
import numpy as np
import os,json,pickle
from scipy.signal import butter, lfilter



def butter_lowpass(cutoff, fs, order):
    """
    Designs a lowpass Butterworth filter

    cutoff: the cutoff frequency of the lowpass filter
    fs:     the sampling frequency used to sample the signal
    order:  the order of the low pass filter

    b,a:    the filter coefficients of the designed lowpass filter
    """
    fny = fs/2
    normal_cutoff = cutoff / fny
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def filteringAndWindowing(data,fs,windowSize,slide,cutoff,order):
    """
    Filters an entire sequence with a lowpass butterworth filter.
    Afterwards the sequence is windowed into smaller trials with 
    an overlap depending on the slide parameter

    data:       The json data containing the signals
    fs:         The sampling rate used to sample the signal
    windowSize: The windowSize used to window the sequences
    slide:      The number of samples the window is slided every time
    cutoff:     The cutoff frequency of the used filter
    order:      The order of the filter

    newTrials:  The resulting trials obtained from filtering and windowing each sequence.

    """

    b,a = butter_lowpass(cutoff = cutoff ,fs=fs,order = order)
    
    data = np.array(data)
    data = lfilter(b,a,data,axis=0)
    newTrials = []
    nSamples = len(data)
    start = 0
    end = windowSize

    while end < nSamples:
        newTrials.append(data[start:end,:])
        start += slide
        end += slide
    
    return newTrials

def loadFileApplyfilterAndSlidingWindow(windowSize,slide,cutoff,order):
    """
    This function loads in the jsonData, filters all the sequences with a lowpass Butterworth filter
    and windows every trial.

    windowSize: the length of the window used
    slide: how many samples the window should move.
    cutoff: the cutoff frequency of the butterworth filter
    order: the order of the lowpass Butterworth filter
    """

    with open("data.json","r") as fin:
        jsonData = json.load(fin)

    fs = 32
    newJson = {}
    for key in jsonData.keys():
        category = jsonData[key]
        newCategory = []
        for f in category:
            newCategory.extend(filteringAndWindowing(f,fs,windowSize,slide,cutoff,order))
        if key[-5:] == "MODEL":
            print("not gonna use " + str(key))
        else:    
            newJson[key] = newCategory

    labeledAccelerometerData ={}
    labeledAccelerometerData["labels"] = []
    labeledAccelerometerData["trials"] = []
    for key, data in newJson.items():
        for array in data:
            labeledAccelerometerData["labels"].append(key)
            labeledAccelerometerData["trials"].append(array.tolist())

    return  labeledAccelerometerData
    

