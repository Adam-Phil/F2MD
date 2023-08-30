import os
import numpy as np
import pickle

savePath = os.getcwd() + "/F2MD/machine-learning-server/saveFile/saveFile_D60_Legacy_V1"
curDateStr = "SVM_together"

def saveData(valuesData, targetData):
    with open(savePath+'/valuesSave_'+curDateStr+'.listpkl', 'wb') as fp:
        pickle.dump(valuesData, fp)
    with open(savePath+'/targetSave_'+curDateStr +'.listpkl', 'wb') as ft:
        pickle.dump(targetData, ft)

def loadData(file_path):
    with open (savePath+'/'+file_path, 'rb') as fp:
        data = pickle.load(fp)
    data = np.array(data)
    return data

def appendData(data, newDataPath):
    with open (savePath+'/'+newDataPath, 'rb') as fp:
        dataAppend = pickle.load(fp)
    dataAppend = np.array(dataAppend)
    retValues = np.append(data,dataAppend,axis = 0)
    return retValues



def main():
    valuesData = None
    targetData = None
    dataFileNames = os.listdir(savePath)
    for fileName in dataFileNames:
        if fileName[-8:] == ".listpkl":
            if "valuesSave" in fileName:
                if valuesData == None:
                    valuesData = loadData(fileName)
                else:
                    appendData(valuesData,fileName)
            elif "targetSave" in fileName:
                if targetData == None:
                    targetData = loadData(fileName)
                else:
                    appendData(targetData,fileName)
            else:
                raise ValueError
    print(valuesData.shape)
    print(targetData.shape)
    saveData(valuesData,targetData)
    



if __name__ == "__main__": main()