from sklearn.neural_network import MLPClassifier
import numpy as np
import joblib
import pickle
import sys
from sklearn.model_selection import GridSearchCV


def rec_round(x):
    s = np.shape(x)
    if len(s) == 1:
        (l,) = s
        ret = np.zeros_like(x)
        for i in range(l):
            rounded = round(x[i], 4)
            ret[i] = rounded
        return ret
    else:
        ret = np.zeros_like(x)
        for i in range(len(x)):
            ret[i] = rec_round(x[i])
        return ret


train_len = 2 / 3

if __name__ == "__main__":
    savePath = "/F2MD/machine-learning-server/saveFile"
    values_path = savePath + "/concat_data/valuesSave_MLP_L1N25_Catch_0.0.listpkl"
    targets_path = savePath + "/concat_data/targetSave_MLP_L1N25_Catch_0.0.listpkl"
    print("Starting...")
    with open(values_path, "rb") as fp:
        X = pickle.load(fp)
    X = np.array(X)
    with open(targets_path, "rb") as ft:
        y = pickle.load(ft)
    y = np.array(y)
    len_seq = float(y.shape[0])
    train_size = int(train_len * len_seq + 0.5)
    order = np.array([i for i in range(int(len_seq))])
    np.random.shuffle(order)
    y = np.take(y, order)
    if len(X.shape) == 3:
        X[[i for i in range(len(order))], :, :] = X[[order], :, :]
        X_train = X[:train_size, :, :]
        X_test = X[train_size:, :, :]
    else:
        X[[i for i in range(len(order))], :] = X[[order], :]
        X_train = X[:train_size, :]
        X_test = X[train_size:, :]
    y_train = y[:train_size]
    y_test = y[train_size:]
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)
    print("--------------------X_train--------------------")
    print(rec_round(X_train[:5]))
    print("--------------------y_train--------------------")
    print(y_train[:5])
    print("--------------------X_test--------------------")
    print(rec_round(X_test[:5]))
    print("--------------------y_test--------------------")
    print(y_test[:5])
    clf = MLPClassifier(activation="logistic")
    layer_sizes = [10,25,50,100]
    one_layer = [(i,) for i in layer_sizes]
    three_layer = [(i,j,k,) for i in layer_sizes for j in layer_sizes for k in layer_sizes]
    five_layer = [(i,j,k,l,m) for i in layer_sizes for j in layer_sizes for k in layer_sizes for l in layer_sizes for m in layer_sizes]
    whole_layers = one_layer + three_layer + five_layer
    parameter_space = {
        "max_iter": [10,50,100,500,1000,5000,10000],
        "hidden_layer_sizes": whole_layers,
        "activation":["identity", "logistic", "tanh", "relu"],
        "solver": ["sgd", "adam", "lbfgs"],
        "alpha": [0.1 / float(i) for i in layer_sizes],
        "learning_rate": ["constant", "adaptive", "invscaling"],
    }
    cv = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=6)
    print("Starting machine learning: MLP_L1N25")
    cv.fit(X, y)
    print('Best parameters found:\n', cv.best_params_)
    means = cv.cv_results_['mean_test_score']
    stds = cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, cv.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
