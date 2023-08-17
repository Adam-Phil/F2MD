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

import os
import json
import numpy as np
from tqdm import tqdm
import pickle

class MlDataCollector:
	def __init__(self, *args, **kwargs):
		print("initializing data collector")
		self.initValuesData = False
		self.valuesData = []
		self.targetData = []

		self.curDateStr = ''

		self.savePath = ''

	def setCurDateSrt(self, datastr):
		self.curDateStr = datastr

	def setSavePath(self, datastr):
		self.savePath = datastr
	
	def saveData(self):
		# print("Values Data to save: " + str(self.valuesData))
		# print("Target Data to save: " + str(self.targetData))
		with open(self.savePath+'/valuesSave_'+self.curDateStr+'.listpkl', 'wb') as fp:
			pickle.dump(self.valuesData, fp)
		with open(self.savePath+'/targetSave_'+self.curDateStr +'.listpkl', 'wb') as ft:
			pickle.dump(self.targetData, ft)

	def loadData(self):
		with open (self.savePath+'/valuesSave_'+self.curDateStr+'.listpkl', 'rb') as fp:
			self.valuesData = pickle.load(fp)
		with open (self.savePath+'/targetSave_'+self.curDateStr +'.listpkl', 'rb') as ft:
			self.targetData = pickle.load(ft)

		# print("Values Data after loading: " + str(self.valuesData))
		# print("Target Data after loading: " + str(self.targetData))


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
		# print("Values Data: " + str(self.valuesData))
		# print("Target Data: " + str(self.targetData))

