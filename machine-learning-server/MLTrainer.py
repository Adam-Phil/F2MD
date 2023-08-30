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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import numpy as np
import joblib
import copy as cp

from keras.models import Sequential  
from keras.layers import Dense, LSTM
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class MlTrainer:

    AIType = "NotSet"

    SavedModel = None
    SavedModelSet = False

    savePath = ''

    curDateStr = ''
    save_y = None
    batch_size = 512
    train_split = 0.8

    cur_index = -1

    newValuesData = []
    newTargetData = []

    def setCurDateSrt(self, datastr):
        self.curDateStr = datastr

    def setSavePath(self, datastr):
        self.savePath = datastr

    def setAIType(self, datastr):
        self.AIType = datastr

    def setSavedModel(self, SavedModel):
        self.SavedModel = SavedModel
        self.SavedModelSet = True

    def train(self, data, le, startTraining):
        self.saveData = cp.deepcopy(data)
        print('Training: ' + self.AIType)
        if (self.AIType != "SVM_SINGLE"):
            filesNames = [f for f in listdir(self.savePath) if isfile(join(self.savePath, f))]
            clf = None
            for s in filesNames:
                if s.startswith('clf_'+ self.AIType) and s.endswith(".pkl"):
                    clf = joblib.load(self.savePath+'/'+s)
            if(self.AIType == "LSTM_RECURRENT"):            
                X,y,d_weights = self.get_values_for_train(data, le, startTraining)
                self.save_y = cp.deepcopy(y)          
                if clf == None:
                    clf = Sequential()
                    clf.add(LSTM(128, return_sequences=True, input_shape=(len(X[0]), len(X[0][0]))))
                    clf.add(LSTM(128, return_sequences=True))
                    clf.add(LSTM(128, return_sequences=False))
                    clf.add(Dense(y.shape[1],activation='softmax'))  
                    clf.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
                reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.7,patience=4, min_lr=0.0005)
                clf.fit(X, y,epochs=10, batch_size=64,class_weight=d_weights, callbacks=[reduce_lr])
            elif(self.AIType == "MLP_SINGLE_L1N25"):
                X,y,d_weights = self.get_values_for_train(data, le, startTraining)
                self.save_y = cp.deepcopy(y)
                if clf == None:
                    clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(25,), random_state=1)
                clf.partial_fit(X, y, classes = list(le.dict_labels.keys()))
            elif(self.AIType == "MLP_SINGLE_L3N25"):
                X,y,d_weights = self.get_values_for_train(data, le, startTraining)
                self.save_y = cp.deepcopy(y)
                if clf == None:
                    clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(25,25,25,), random_state=1)
                clf.partial_fit(X, y, classes = list(le.dict_labels.keys()))
        else:
            X,y,d_weights = self.get_values_for_train(data, le, startTraining)
            self.save_y = cp.deepcopy(y) 
            clf = SVC(gamma=0.001, C=100.)
            clf.fit(X, y)

        joblib.dump(clf, self.savePath + '/clf_'+self.AIType + '_'+self.curDateStr+'.pkl')
        
        print("Saved " + self.savePath + '/clf_'+self.AIType + '_'+self.curDateStr+'.pkl')

        print(data.valuesData)
        if "LSTM" in self.AIType:
            y_test = clf.predict(X,batch_size=self.batch_size)
            pred=np.argmax(y_test,axis=1)
            ground=data.targetData
        else:
            y_test = clf.predict(data.valuesData)
            pred = y_test
            ground= data.targetData
        
        print("Predicting Finished!")

        res=classification_report(ground, pred)
        print(le.dict_labels)
        print(res)
        conf_mat = confusion_matrix(ground, pred)
        fig, ax = plt.subplots(figsize=(14,14))
        sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=le.classes_, yticklabels=le.classes_)
        plt.ylabel('ground truth')
        plt.xlabel('prediction')
        # plt.show()
        plt.savefig(self.savePath+'/fig_'+self.AIType + '_'+self.curDateStr+'.png', dpi=fig.dpi)
        plt.close()

    def get_d_weights(self, y,le):
        y_train_str = le.inverse_transform(y)
        le_classes_remove = []
        for i in range(0,len(le.classes_)):
            if le.classes_[i] not in y_train_str:
                le_classes_remove.append(i)
        le_classes_temp = np.delete(le.classes_, le_classes_remove)
        w = compute_class_weight(class_weight='balanced', classes=list(le_classes_temp), y=y_train_str)
        new_w = np.array([])
        int_i = 0
        for i in range(0,len(le.classes_)):
            if le.classes_[i] not in le_classes_temp:
                new_w = np.append(new_w,[1])
            else:
                new_w = np.append(new_w,w[int_i])
                int_i = int_i + 1
        d_weights = dict(enumerate(new_w))
        gen_index = le.transform(['Genuine'])[0]
        d_weights[gen_index] = (len(le.classes_)-1) * d_weights[gen_index]
        # print("d_weights" + str(d_weights))
        return d_weights
    
    def get_values_for_train(self, data, le, startTraining):
        
        # y part
        print("---------Y Part---------")
        y = np.array(data.targetData)
        y = np.reshape(y, (1,np.product(y.shape)))[0]
        y = np.array(y[startTraining:len(y)])
        # print(y)
        d_weights = self.get_d_weights(y, le)
        if "LSTM" in self.AIType: # Don't know if this is even needed
           y = to_categorical(y)  
        print(y)

        # x part
        print("----------X Part----------")
        if False and "LSTM" in self.AIType:
            X = data.valuesData
        else: 
            X = np.array(data.valuesData)
            X = np.array(X[startTraining:len(X),:])
            print(len(data.valuesData))
            print(X.shape)
            print(y.size)
        return X,y,d_weights

