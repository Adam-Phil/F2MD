import os
import numpy as np
import pickle

savePath = os.getcwd() + "/F2MD/machine-learning-server/saveFile/saveFile_D60_Legacy_V1"
curDateStr = "SVM_together"

def saveData(valuesData, targetData):
    with open(savePath+'/valuesSave_'+curDateStr+'.listpkl', 'ab') as fp:
        pickle.dump(valuesData, fp)
    fp.close()
    with open(savePath+'/targetSave_'+curDateStr +'.listpkl', 'ab') as ft:
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
    dataFileNames = os.listdir(savePath)
    for fileName in dataFileNames:
        if fileName[-8:] == ".listpkl" and not "together" in fileName:
            print("Saving: " + fileName)
            if "valuesSave" in fileName:
                if firstValue:
                    valuesData = loadData(fileName)
                    firstValue = False
                else:
                    appendData(valuesData,fileName)
                print(valuesData.shape)
            elif "targetSave" in fileName:
                if firstTarget:
                    targetData = loadData(fileName)
                    firstTarget = False
                else:
                    appendData(targetData,fileName)
                print(targetData.shape)
            else:
                raise ValueError
    saveData(valuesData,targetData)
    controlValues = loadData("valuesSave_SVM_together.listpkl")
    controlTargets = loadData("targetSave_SVM_together.listpkl")
    print(controlValues.shape)
    print(controlTargets.shape)

if __name__ == "__main__": main()