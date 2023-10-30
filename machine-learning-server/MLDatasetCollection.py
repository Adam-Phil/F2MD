import os
import numpy as np
import pickle
from tqdm import tqdm
import sys


def model_name_to_short(model_name):
    if "SVM" in model_name:
        return "SVM"
    elif "MLP" in model_name:
        if "L1N25" in model_name:
            return "MLP_L1N25"
        elif "L3N25" in model_name:
            return "MLP_L3N25"
        else:
            raise ValueError("Not suitable model")
    elif "LSTM" in model_name:
        return "LSTM"
    else:
        raise ValueError("Not suitable model")
    
def model_name_to_number(model_name):
    if "SVM" in model_name:
        return 0
    elif "MLP" in model_name:
        if "L1N25" in model_name:
            return 1
        elif "L3N25" in model_name:
            return 2
        else:
            raise ValueError("Not suitable model")
    elif "LSTM" in model_name:
        return 3
    else:
        raise ValueError("Not suitable model")

def number_to_model_name(model):
    if model == 0:
        return "SVM"
    elif model == 1:
        return "MLP_L1N25"
    elif model == 2:
        return "MLP_L3N25"
    elif model == 3:
        return "LSTM"
    else:
        raise ValueError("Not suitable model")


def determineCurrentCheckVersion(checkVersion):
    if checkVersion == 0:
        return "Legacy"
    elif checkVersion == 1:
        return "Catch"
    elif checkVersion == 2:
        return "Experi"
    else:
        raise ValueError("Unknown Checks Version")


def saveData(valuesData, targetData, partition, savePath, currentModel, checkVersion):
    with open(
        savePath
        + "/valuesSave_"
        + currentModel
        + "_"
        + checkVersion
        + "_"
        + str(partition)
        + ".listpkl",
        "wb",
    ) as fp:
        pickle.dump(valuesData, fp)
    fp.close()
    with open(
        savePath
        + "/targetSave_"
        + currentModel
        + "_"
        + checkVersion
        + "_"
        + str(partition)
        + ".listpkl",
        "wb",
    ) as ft:
        pickle.dump(targetData, ft)
    ft.close()


def loadData(file_path, savePath):
    with open(savePath + "/" + file_path, "rb") as f:
        data = pickle.load(f)
    f.close()
    data = np.array(data)
    return data


def appendData(data, newDataPath, savePath):
    with open(savePath + "/" + newDataPath, "rb") as f:
        dataAppend = pickle.load(f)
    f.close()
    dataAppend = np.array(dataAppend)
    retValues = np.append(data, dataAppend, axis=0)
    return retValues


def main():
    print("Collecting Data")
    partition_numbers = 10
    # partitions = [(h+1)/partition_numbers for h in range(partition_numbers)]
    partitions = [1.0]

    model = sys.argv[1]
    
    savePath = "/F2MD/machine-learning-server"
    if model.isdigit():
        currentModel = number_to_model_name(int(model))
    else:
        currentModel = model_name_to_short(model)
    if "SVM" in currentModel:
        partitions = [round(elem / 5, 2) for elem in partitions]
    checkVersion = determineCurrentCheckVersion(int(sys.argv[2]))
    loadPath = (
        savePath
        + "/saveFile/data/"
        + checkVersion
        + "_Checks_Data/"
        + currentModel
    )
    # loadPath = savePath + "/saveFile/" + currentModel
    savePath = savePath + "/saveFile/concat_data"
    print(savePath)
    print(loadPath)
    firstValue = True
    firstTarget = True
    dataFileNames = [
        elem
        for elem in os.listdir(loadPath)
        if (elem[-8:] == ".listpkl" and not "together" in elem)
    ]
    valuesFileNames = [elem for elem in dataFileNames if "values" in elem]
    targetFileNames = [elem for elem in dataFileNames if "target" in elem]
    # print(len(targetFileNames))
    valuesNamesLength = len(valuesFileNames)
    partition_sizes = [
        int(round(partitions[j] * valuesNamesLength, 0)) for j in range(len(partitions))
    ]
    # print(partition_sizes)
    for i in tqdm(range(valuesNamesLength)):
        valuesFileName = valuesFileNames[i]
        targetFileName = targetFileNames[i]
        if firstValue:
            # print("Saving: " + valuesFileName)
            valuesData = loadData(valuesFileName, loadPath)
            firstValue = False
        else:
            valuesData = appendData(valuesData, valuesFileName, loadPath)
        # print(valuesData.shape)
        if firstTarget:
            # print("Saving: " + targetFileName)
            targetData = loadData(targetFileName, loadPath)
            firstTarget = False
        else:
            targetData = appendData(targetData, targetFileName, loadPath)
        # print(targetData.shape)
        if model == 1:
            rounded_partition = round((i + 1) / valuesNamesLength, 2)
        else:
            rounded_partition = round((i + 1) / valuesNamesLength, 1)
        # print(rounded_partition)
        if (i + 1) in partition_sizes:
            saveData(
                valuesData,
                targetData,
                rounded_partition,
                savePath,
                currentModel,
                checkVersion,
            )
            partitions.pop(0)
            partition_sizes.pop(0)
            # print(partitions)
    print("Values Data Shape: " + str(valuesData.shape))


if __name__ == "__main__":
    main()
