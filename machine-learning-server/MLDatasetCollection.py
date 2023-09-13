import os
import numpy as np
import pickle
from tqdm import tqdm

savePath = os.getcwd() + "/machine-learning-server/saveFile/saveFile_D60_Legacy_V1"
currentModel = "LSTM"
partition = 0.5

def saveData(valuesData, targetData):
    with open(savePath+'/valuesSave_'+currentModel + "_together_" + str(partition) +'.listpkl', 'wb') as fp:
        pickle.dump(valuesData, fp)
    fp.close()
    with open(savePath+'/targetSave_'+currentModel + "_together_" + str(partition) +'.listpkl', 'wb') as ft:
        pickle.dump(targetData, ft)
    ft.close()

def loadData(file_path):
    with open (savePath+'/'+file_path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    data = np.array(data)
    return data

def appendData(data, newDataPath):
    with open (savePath+'/'+newDataPath, 'rb') as f:
        dataAppend = pickle.load(f)
    f.close()
    dataAppend = np.array(dataAppend)
    retValues = np.append(data,dataAppend,axis = 0)
    return retValues



def main():
    firstValue = True
    firstTarget = True
    dataFileNames = [elem for elem in os.listdir(savePath) if (elem[-8:] == ".listpkl" and not "together" in elem)]
    valuesFileNames = [elem for elem in dataFileNames if "values" in elem]
    targetFileNames = [elem for elem in dataFileNames if "target" in elem]
    # print(len(targetFileNames))
    for i in tqdm(range(int(len(valuesFileNames)*partition))):
        valuesFileName = valuesFileNames[i]
        targetFileName = targetFileNames[i]
        if firstValue:
            # print("Saving: " + valuesFileName)
            valuesData = loadData(valuesFileName)
            firstValue = False
        else:
            valuesData = appendData(valuesData,valuesFileName)
        # print(valuesData.shape)
        if firstTarget:
            # print("Saving: " + targetFileName)
            targetData = loadData(targetFileName)
            firstTarget = False
        else:
            targetData = appendData(targetData,targetFileName)
        # print(targetData.shape)
    saveData(valuesData,targetData)
    controlValues = loadData("valuesSave_" + currentModel + "_together_" + str(partition) + ".listpkl")
    controlTargets = loadData("targetSave_" + currentModel + "_together_"+ str(partition) + ".listpkl")
    print(controlValues.shape)
    print(controlTargets.shape)

if __name__ == "__main__": main()