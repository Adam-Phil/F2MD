import os
import numpy as np
import pickle

savePath = os.getcwd() + "/F2MD/machine-learning-server/saveFile/saveFile_D60_Legacy_V1"
curDateStr = "2023-08-30_14:55:56"

def saveData(valuesData, targetData):
    with open(savePath+'/valuesSave_'+curDateStr+'.listpkl', 'wb') as fp:
        pickle.dump(valuesData, fp)
    with open(savePath+'/targetSave_'+curDateStr +'.listpkl', 'wb') as ft:
        pickle.dump(targetData, ft)

def loadData(number):
    with open (savePath+'/valuesSave_'+str(number)+'.listpkl', 'rb') as fp:
        valuesData = pickle.load(fp)
    with open (savePath+'/targetSave_'+str(number) +'.listpkl', 'rb') as ft:
        targetData = pickle.load(ft)
    valuesData = np.array(valuesData)
    targetData = np.array(targetData)
    return valuesData, targetData

def appendData(valuesData, targetData, numberNew):
    with open (savePath+'/valuesSave_'+str(numberNew)+'.listpkl', 'rb') as fp:
        valuesDataAppend = pickle.load(fp)
    with open (savePath+'/targetSave_'+str(numberNew) +'.listpkl', 'rb') as ft:
        targetDataAppend = pickle.load(ft)
    valuesDataAppend = np.array(valuesDataAppend)
    targetDataAppend = np.array(targetDataAppend)
    retValues = np.append(valuesData,valuesDataAppend, axis = 0)
    retTargets = np.append(targetData,targetDataAppend, axis = 0)
    return retValues, retTargets



def main():
    valuesData,targetData = loadData(1)
    newVals, newTargs = appendData(valuesData, targetData, 2)
    print(newVals.shape)
    print(newTargs.shape)
    



if __name__ == "__main__": main()