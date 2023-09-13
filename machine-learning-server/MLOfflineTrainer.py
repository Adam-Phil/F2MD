from sklearn.svm import SVC

import numpy as np
import joblib
import pickle
import os

partition = 0.1
savePath = os.getcwd() + "/machine-learning-server/saveFile/saveFile_D60_Legacy_V1" #TODO: Give the right save path here, if this is not right

def loadData(file_path):

    with open (savePath+'/'+file_path, 'rb') as fp:
        data = pickle.load(fp)
    data = np.array(data)
    return data

def main():
    X = loadData("valuesSave_SVM_together_" + str(partition) + ".listpkl")
    y = loadData("targetSave_SVM_together_" + str(partition) + ".listpkl")
    clf = SVC(gamma=0.001, C=100., verbose=1)
    clf.fit(X, y)
    joblib.dump(clf, savePath + '/clf_'+ "SVM_togehter_" + str(partition) + '.pkl')

if __name__ == "__main__": main()