"""
/*******************************************************************************
 * @author  Joseph Kamel
 * @email   josephekamel@gmail.com
 * @date	28/11/2018
 * @version 2.0
 *
 * SCA (Secure Cooperative Autonomous systems)
 * Copyright (c) 2013, 2018 Institut de Recherche Technologique SystemX
 * All rights reserved.
 *******************************************************************************/
"""

import pickle
import datetime
import os

def deepMkDir(path):
	splitted_path = path.split("/")
	if path.startswith("/"):
		existent_path = ""
	elif path.startswith("."):
		existent_path = "."
	else:
		raise ValueError("Please give a right path for deepMkDir")
	for single in splitted_path:
		existent_path = existent_path + "/" + single
		if not (os.path.exists(existent_path) and os.path.isdir(existent_path)):
			os.mkdir(existent_path)

class MlDataCollector:
	def __init__(self):
		print("initializing data collector")
		self.initValuesData = False
		self.valuesData = []
		self.targetData = []

		self.curDateStr = ''
		self.AIType = ""
		self.savePath = ''

	def setCurDateSrt(self, datastr):
		self.curDateStr = datastr

	def setSavePath(self, datastr):
		self.savePath = datastr
	
	def setAIType(self, AIType):
		if "SVM" in AIType:
			self.AIType = "SVM"
		elif "LSTM" in AIType:
			self.AIType = "LSTM"
		elif "MLP" in AIType:
			if "L1N25" in AIType:
				self.AIType = "MLP_L1N25"
			elif "L3N25" in AIType:
				self.AIType = "MLP_L3N25"
			else:
				raise ValueError("Unknown MLP type")
		else:
			raise ValueError("Unknown AIType")

	def saveData(self, check_type):
		# print("Values Data to save: " + str(self.valuesData))
		# print("Target Data to save: " + str(self.targetData))
		complete_save_path = self.savePath+"/data/" + check_type +'_Checks_Data/' + self.AIType
		if not (os.path.exists(complete_save_path) and os.path.isdir(complete_save_path)):
			deepMkDir(complete_save_path)
		self.curDateStr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
		with open(complete_save_path + '/valuesSave_'+self.curDateStr+'.listpkl', 'wb') as fp:
			pickle.dump(self.valuesData, fp)
		with open(complete_save_path + '/targetSave_'+self.curDateStr +'.listpkl', 'wb') as ft:
			pickle.dump(self.targetData, ft)
		self.valuesData = []
		self.targetData = []

	def prepare_arrays(self):
		if isinstance(self.valuesData[0], list):
			for i in range(0,len(self.valuesData)):
				self.valuesData[i] = self.valuesData[i]
		else:
			self.valuesData = self.valuesData
		self.targetData = self.targetData

	def collectData(self,bsmArray):
		if not self.initValuesData:
			self.initValuesData = True
			if isinstance(bsmArray[0], list):
				for i in range(0,len(bsmArray[0])):
					self.valuesData.append([])
		if isinstance(bsmArray[0], list):
			for i in range(0,len(bsmArray[0])):
				self.valuesData[i].append(bsmArray[0][i])
		else:
			self.valuesData.append(bsmArray[0])
		self.targetData.append(bsmArray[1])
		# print("Collecting Data done!")
		# print("Values Data: " + str(self.valuesData))
		# print("Target Data: " + str(self.targetData))

