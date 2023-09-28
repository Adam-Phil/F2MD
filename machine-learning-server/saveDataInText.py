import os
import pickle
import numpy as np

if __name__ == "__main__":
    path = "/home/philip/auslagerung/MLP_L1N25/"
    target_data = [clf for clf in os.listdir(path) if ("target" in clf and clf.endswith(".listpkl"))]
    values_data = [clf for clf in os.listdir(path) if ("value" in clf and clf.endswith(".listpkl"))]
    one_t = target_data[74]
    one_v = values_data[74]
    with open (path+one_v, 'rb') as fp:
        X = pickle.load(fp)
    X = np.array(X)
    with open (path+one_t, 'rb') as ft:
        y = pickle.load(ft)
    y = np.array(y)
    with open("/home/philip/f2md-training/F2MD/machine-learning-server/saveFile/sampleData", "w") as data:
        for i in range(1000):
            y1 = y[i]
            x1 = X[i]
            for j in range(len(x1)):
                x11 = x1[j]
                if x11 < 0 or x11 > 1:
                    print("Help normalize")

