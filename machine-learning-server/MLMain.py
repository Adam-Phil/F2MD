"""
/*******************************************************************************
 * @author  Joseph Kamel
 * @email   josephekamel@gmail.com
 * @date    28/11/2018
 * @version 2.0
 *
 * SCA (Secure Cooperative Autonomous systems)
 * Copyright (c) 2013, 2018 Institut de Recherche Technologique SystemX
 * All rights reserved.
 *******************************************************************************/
"""

from os import listdir
from os.path import isfile, join
import os
import json
import numpy as np
from MLDataCollector import MlDataCollector, deepMkDir
from MLNodeStorage import MlNodeStorage
from MLTrainer import MlTrainer
import datetime
import joblib

from MLLabelEncoder import MlLabelEncoder
from MLStats import MlStats
from MLVarThresholdLite import MlVarThresholdLite

import time

clf = None
RTtrain = False
RTsave = True

time_based_save = True
lower_bounds = [25200, 50400]
upper_bounds = [32400, 57600]

Positive_Threshold = 0.5

RTFilterTime = 100
RTFilterKeepTime = 600


def saveJson(bsmJsonString):
    with open("sample.json", "w") as file:
        json.dump(bsmJsonString, file)


class MlMain:
    initiated = False

    curDateStr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dataCollector = MlDataCollector()
    trainer = MlTrainer()
    storage = MlNodeStorage()
    arrayLength = 60

    collectDur = 0
    deltaCall = 100000
    trainedSamples = 0

    clf = None
    savePath = "./saveFile"
    check_type = ""

    meanRuntime = 0
    meanRuntime_p = 0
    numRuntime = 0
    printRuntime = 10000 * 10000
    printRuntimeCnt = 0

    filterdelta = 0

    labels_legacy = [
        "Genuine",
        "LocalAttacker",
    ]

    le = MlLabelEncoder()

    stats = MlStats()
    varthrelite = MlVarThresholdLite()

    def setCheckType(self, bsmJsonString):
        self.check_type = bsmJsonString[:-6]
        print("Set Check Type to: " + self.check_type)

    def create_save_folders(self):
        clf_path = self.savePath + "/clfs"
        if not (os.path.exists(clf_path) and os.path.isdir(clf_path)):
            deepMkDir(clf_path)
        data_path = self.savePath + "/data"
        if not (os.path.exists(data_path) and os.path.isdir(data_path)):
            deepMkDir(data_path)
        concat_data_path = self.savePath + "/concat_data"
        if not (os.path.exists(concat_data_path) and os.path.isdir(concat_data_path)):
            deepMkDir(concat_data_path)

    def init(self, AIType):
        self.le.fit(self.labels_legacy)

        self.create_save_folders()

        self.dataCollector.setCurDateSrt(self.curDateStr)
        self.dataCollector.setSavePath(self.savePath)
        self.trainer.setCurDateSrt(self.curDateStr)
        self.trainer.setSavePath(self.savePath)
        self.trainer.setAIType(AIType)
        self.trainedModelExists(AIType)

    def checkData(self, bsmJsom):
        generationTime = bsmJsom["BsmPrint"]["Metadata"]["generationTime"]
        for i in range(len(lower_bounds)):
            low = lower_bounds[i]
            upp = upper_bounds[i]
            if generationTime >= low and generationTime <= upp:
                return True
        return False

    def mlMain(self, bsmJsonString, AIType):
        if not self.initiated:
            self.init(AIType)
            self.initiated = True

        start_time = time.time()

        bsmJsom = json.loads(bsmJsonString)
        # saveJson(bsmJsom)
        curArray = self.getNodeArray(bsmJsom, AIType)
        # print(RTsave)
        if RTsave:
            if self.collectDur < self.deltaCall:
                # print(self.collectDur)
                self.collectDur = self.collectDur + 1
                if self.checkData(bsmJsom):
                    self.dataCollector.collectData(curArray)
            else:
                print("DataSave And Training " + str(self.deltaCall) + " Started ...")
                self.collectDur = 0
                if len(self.dataCollector.valuesData) != 0:
                    self.dataCollector.saveData(self.check_type)
                # self.inventory()
                if RTtrain:
                    # print(len(self.dataCollector.valuesData))
                    self.trainer.train(self.dataCollector, self.le, self.trainedSamples)
                    self.clf = joblib.load(self.savePath + "/clf_" + AIType + ".pkl")
                    self.deltaCall = len(self.dataCollector.valuesData) / 2
                    self.trainedSamples = len(self.dataCollector.valuesData)
                print("DataSave And Training " + str(self.deltaCall) + " Finished!")

        return_value = "False"

        if self.clf is None:
            # print("No clf loaded")
            return_value = "False"
            start_time_p = 0.0
            end_time_p = 0.0
        else:
            # print("Clf judging")
            if "LSTM" in AIType:  # try out if this is necessarry here
                self.clf.reset_states()
            array_npy = np.array([curArray[0]])
            start_time_p = time.time()
            pred_array = self.clf.predict(array_npy)
            end_time_p = time.time()
            gen_index = self.le.transform(["Genuine"])[0]
            if "SVM" in AIType or "MLP" in AIType:
                prediction = pred_array[0]
            else:
                prediction = pred_array[0][1 - gen_index]

            label_index = self.le.transform(
                [bsmJsom["BsmPrint"]["Metadata"]["mbType"]]
            )[0]
            self.varthrelite.update_stats(prediction, label_index)
            if prediction > Positive_Threshold:
                self.stats.update_stats(True, label_index)
                return_value = "True"
            else:
                self.stats.update_stats(False, label_index)
                return_value = "False"
        end_time = time.time()
        self.meanRuntime = (
            self.numRuntime * self.meanRuntime + (end_time - start_time)
        ) / (self.numRuntime + 1)
        self.meanRuntime_p = (
            self.numRuntime * self.meanRuntime_p + (end_time_p - start_time_p)
        ) / (self.numRuntime + 1)
        if self.printRuntimeCnt >= self.printRuntime:
            self.printRuntimeCnt = 0
            print(
                "meanRuntime: "
                + str(self.meanRuntime)
                + " "
                + str(self.numRuntime)
                + " predict:"
                + str(self.meanRuntime_p)
            )
            self.stats.print_stats()
            self.varthrelite.print_stats()
            self.printRuntimeCnt = self.printRuntimeCnt + 1
        else:
            self.printRuntimeCnt = self.printRuntimeCnt + 1
        self.numRuntime = self.numRuntime + 1
        # print("Returning Value: " + return_value)
        return return_value

    def getNodeArray(self, bsmJsom, AIType):
        receiverId = bsmJsom["BsmPrint"]["Metadata"]["receiverId"]
        pseudonym = bsmJsom["BsmPrint"]["BSMs"][0]["Pseudonym"]
        time = bsmJsom["BsmPrint"]["Metadata"]["generationTime"]

        label = bsmJsom["BsmPrint"]["Metadata"]["mbType"]
        if label == "GlobalAttacker":
            label = "Genuine"

        numLabel = np.array(self.le.transform([label])[0], dtype=np.int8)

        self.storage.add_bsm(
            receiverId, pseudonym, time, bsmJsom, self.arrayLength, numLabel
        )

        if time - self.filterdelta > RTFilterTime:
            self.filterdelta = time
            self.storage.filter_bsms(time, RTFilterKeepTime)

        # TODO: Only take SINGLE or FEATURES here, everything else is some kind of historical data, which you do not need
        if (
            "SINGLE" in AIType
        ):  # takes the first 24 features of the arrays (so without position and so on)
            returnArray = self.storage.get_array(receiverId, pseudonym)
        if (
            "FEATURES" in AIType
        ):  # takes the first 18 features so only the plausibility checks
            returnArray = self.storage.get_array_features(receiverId, pseudonym)
        if (
            "AVEFEAT" in AIType
        ):  # takes the average and minimum over the arrayLength messages of array features and adds the last label to it (length 36 and 1)
            returnArray = self.storage.get_array_MLP_features(
                receiverId, pseudonym, self.arrayLength
            )
        if (
            "AVERAGE" in AIType
        ):  # takes the same as before but adds the length of the history and the last positional data of the vehicle
            returnArray = self.storage.get_array_MLP(
                receiverId, pseudonym, self.arrayLength
            )
        if "RECURRENT" in AIType:  # for LSTM as it uses sequential data
            returnArray = self.storage.get_array_lstm(
                receiverId, pseudonym, self.arrayLength
            )
        if "RECUFEAT" in AIType:
            returnArray = self.storage.get_array_lstm_feat(
                receiverId, pseudonym, self.arrayLength
            )
        if "RECUSIN" in AIType:
            returnArray = self.storage.get_array_lstm_sin(
                receiverId, pseudonym, self.arrayLength
            )
        if "RECUMIX" in AIType:
            returnArray = self.storage.get_array_lstm_mix(
                receiverId, pseudonym, self.arrayLength
            )
        if "RECUALL" in AIType:
            returnArray = self.storage.get_array_lstm_all(
                receiverId, pseudonym, self.arrayLength
            )
        if "COMBINED" in AIType:
            returnArray = self.storage.get_array_combined(
                receiverId, pseudonym, self.arrayLength
            )

        # print("cur_array: " + str(cur_array))
        # print("returnArray: " + str(returnArray))
        return returnArray

    def trainedModelExists(self, AIType):
        filesNames = [
            f
            for f in listdir(self.savePath + "/clfs/")
            if isfile(join(self.savePath + "/clfs/", f))
        ]
        print("trainedModelExists?")

        for s in filesNames:
            if s.startswith("clf_" + AIType) and s.endswith(".pkl") and AIType in s:
                print("Loading " + s + " " + AIType + " ...")
                self.clf = joblib.load(self.savePath + "/" + s)
                self.curDateStr = s[-23:-4]
                self.dataCollector.setCurDateSrt(self.curDateStr)
                self.trainer.setCurDateSrt(self.curDateStr)
                # self.dataCollector.loadData() # Maybe works without this
                # self.deltaCall = self.dataCollector.valuesCollection.shape[0]/5
                print(
                    "Loading " + str(len(self.dataCollector.valuesData)) + " Finished!"
                )

        # print(self.clf.coefs_)

    def inventory(self):
        print("----------DataCollector----------")
        print("-----ValuesDataLength: " + str(len(self.dataCollector.valuesData)))
        print("-----TargetDataLength: " + str(len(self.dataCollector.targetData)))
        print("----------NodeStorage----------")
        print("-----IdIndexLength: " + str(len(self.storage.id_index)))
        print("-----IdStorageLength: " + str(len(self.storage.id_storage)))
        indiv_stor_elem_length = []
        for elem in self.storage.id_storage:
            indiv_stor_elem_length.append(len(elem.id_array_y))
        print("-----IndividualIdStorageElementsLengths: " + str(indiv_stor_elem_length))
        print("----------VarThresholdLite----------")
        print("-----List Lengths: " + str(len(self.varthrelite.step_stats)))
