from sklearn.svm import SVC

import numpy as np
import joblib
import pickle

savePath = "saveFile/saveFile_D60_Legacy_V1" #TODO: Give the right save path here, if this is not right

def loadData(file_path):
    with open (savePath+'/'+file_path, 'rb') as fp:
        data = pickle.load(fp)
    data = np.array(data)
    return data

def main():
    X = loadData("valuesSave_SVM_together.listpkl")
    y = loadData("targetSave_SVM_togehter.listpkl")
    clf = SVC(gamma=0.001, C=100.)
    clf.fit(X, y)
    joblib.dump(clf, savePath + '/clf_'+ "SVM_togehter" + '.pkl')

if __name__ == "__main__": main()