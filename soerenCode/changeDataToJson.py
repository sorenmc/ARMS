import pandas as pd
import numpy as np
import os,json

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


base = os.path.dirname(__file__)
dataPath = os.path.abspath(os.path.join(base, "..", "Data"))

dataToJson = {}

for folder in os.scandir(dataPath):
    temp = []
    if folder.is_dir():
        for files in os.listdir(folder):
            filePath = os.path.join(folder, files)
            temp.append(pd.read_csv(filePath,sep=' ').values)
    dataToJson[folder.name] = temp

dataToJson 
dumped = json.dumps(dataToJson, cls=NumpyEncoder)

with open("data.json","w") as f:
    f.write(dumped)

        