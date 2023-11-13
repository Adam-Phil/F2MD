from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from MLDatasetCollection import model_name_to_number
import tensorflow as tf
import numpy as np
import joblib
import pickle
import sys
import random


random.seed(10)
np.random.seed(10)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


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


def pre_process(X, y, zero_add):
    print("--------------------Preprocessing Data--------------------")
    number_of_ones = sum(y)
    should_zeros = int(number_of_ones * (1.0 + zero_add))
    indexes_ones = np.array([i for i in range(len(y)) if y[i] == 1])
    other_indexes = np.array([i for i in range(len(y)) if not (i in indexes_ones)])
    np.random.shuffle(other_indexes)
    other_indexes = other_indexes[:should_zeros]
    y = y[indexes_ones.tolist() + other_indexes.tolist(),]
    X = X[indexes_ones.tolist() + other_indexes.tolist(),]
    print("--------------------Preprocessing Done--------------------")
    return X, y


def print_maxes(X):
    arr = [i for i in range(10)]
    print(arr[:-6])
    ma = np.amax(X, axis=0)
    mi = np.amin(X, axis=0)
    print("Maximums: " + str(ma))
    print("Minimums: " + str(mi))


def sort_to_class(y):
    ret = []
    for i in range(len(y)):
        one_pred = y[i]
        if one_pred[0] > one_pred[1]:
            ret.append(0)
        else:
            ret.append(1)
    return ret


train_len = 2 / 3
print_data = False
print_classes_and_coefs = False
print_maxims = False
min_max_scale = True
standard_scale = False
leave_out_last6 = False
zero_add = 5
every_tenth = False

if __name__ == "__main__":
    model = sys.argv[1]
    savePath = "/F2MD/machine-learning-server/saveFile"
    values_path = savePath + "/concat_data/valuesSave_MLP_L1N25_Catch_0.1.listpkl"
    targets_path = savePath + "/concat_data/targetSave_MLP_L1N25_Catch_0.1.listpkl"
    print("Starting...")
    if model.isdigit():
        model = int(model)
    else:
        model = model_name_to_number(model_name=model)
    with open(values_path, "rb") as fp:
        X = pickle.load(fp)
    X = np.array(X)
    with open(targets_path, "rb") as ft:
        y = pickle.load(ft)
    y = np.array(y)
    class_weight = np.flip(compute_class_weight("balanced", classes=[0, 1], y=y))
    if print_maxims:
        print_maxes(X)
    if min_max_scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    if standard_scale:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    if leave_out_last6:
        X = X[:, :-6]
    if every_tenth:
        tenth_list = [i * 10 for i in range(int(len(y) / 10))]
        X_train = X[tenth_list, :]
        y_train = y[tenth_list]
        len_seq = float(y.shape[0])
        train_size = int(train_len * len_seq + 0.5)
        order = np.array([i for i in range(int(len_seq))])
        np.random.shuffle(order)
        y = np.take(y, order)
        X[[i for i in range(len(order))], :] = X[[order], :]
        X_test = X[train_size:, :]
        y_test = y[train_size:]
    else:
        len_seq = float(y.shape[0])
        train_size = int(train_len * len_seq + 0.5)
        order = np.array([i for i in range(int(len_seq))])
        np.random.shuffle(order)
        y = np.take(y, order)
        X[[i for i in range(len(order))], :] = X[[order], :]
        X_train = X[:train_size, :]
        X_test = X[train_size:, :]
        y_train = y[:train_size]
        y_test = y[train_size:]
    if zero_add != None:
        X_train, y_train = pre_process(X_train, y_train, zero_add)
    if print_data:
        print("--------------------X_train--------------------")
        print(rec_round(X_train[:5]))
        print("--------------------y_train--------------------")
        print(y_train[:5])
        print("--------------------X_test--------------------")
        print(rec_round(X_test[:5]))
        print("--------------------y_test--------------------")
        print(y_test[:5])
    if model == 0:
        clf = SVC(gamma=0.001, C=100.0, verbose=1)
        print("Starting machine learning: SVM")
        clf.fit(X_train, y_train)
        print("Machine Learning done; saving model")
        #joblib.dump(clf, savePath + "/clfs/clf_SVM_SINGLE" + ".pkl")
    elif model == 1:
        clf = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100),n_iter_no_change=1000,max_iter=15000, verbose=True, solver="adam", random_state=1)
        print("Starting machine learning: MLP_L1N25")
        clf.fit(X_train, y_train)
        print("Machine Learning done; saving model")
        #joblib.dump(clf, savePath + "/clfs/clf_MLP_SINGLE_L1N25" + ".pkl")
    elif model == 2:
        clf = MLPClassifier(
            alpha=1e-5,
            hidden_layer_sizes=(25, 25, 25),
            solver="lbfgs",
            activation="relu",
            random_state=1,
            n_iter_no_change=100,
            max_iter=1000,
        )
        print("Starting machine learning: MLP_L3N25")
        clf.fit(X_train, y_train)
        print("Machine Learning done; saving model")
        #joblib.dump(clf, savePath + "/clfs/clf_MLP_SINGLE_L3N25" + ".pkl")
    elif model == 3:
        y_train = to_categorical(y_train)
        print(X_train.shape)
        print(y_train.shape)
        print("Data loaded")
        clf = Sequential()
        clf.add(
            LSTM(
                128,
                return_sequences=True,
                input_shape=(len(X_train[0]), len(X_train[0][0])),
            )
        )
        clf.add(LSTM(128, return_sequences=True))
        clf.add(LSTM(128, return_sequences=False))
        clf.add(Dense(y_train.shape[1], activation="sigmoid"))
        clf.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        reduce_lr = ReduceLROnPlateau(
            monitor="accuracy", factor=0.7, patience=4, min_lr=0.0005
        )
        clf.fit(X_train, y_train, epochs=10, batch_size=64, callbacks=[reduce_lr])
        print("Starting machine learning: LSTM")
        print("Machine Learning done; saving model")
        #joblib.dump(clf, savePath + "/clfs/clf_LSTM_RECURRENT" + ".pkl")
    elif model == 4:
        clf = Sequential()
        clf.add(Dense(100, input_shape=(len(X_train[0]),)))
        clf.add(Dense(2, activation="sigmoid"))

        y_train = to_categorical(y_train)
        clf.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            loss_weights=class_weight,
            metrics=["accuracy"],
        )
        clf.fit(X_train, y_train)
        tf.keras.utils.plot_model(clf)
        #joblib.dump(clf, savePath + "/clfs/clf_MLP_Weighted" + ".pkl")
    else:
        raise ValueError("Not a suitable model selected")
    if print_classes_and_coefs:
        print("Classes: " + str(clf.classes_) + "\n")
        print("Loss: " + str(round(clf.loss_, 4)) + "\n")
        for i in range(len(clf.coefs_)):
            print("----------Coef " + str(i) + ":----------\n")
            coef = clf.coefs_[i]
            print(str(coef) + "\n")
    y_test_pred = clf.predict(X_test)
    if model == 4:
        y_test_pred = sort_to_class(y_test_pred)
    true_n_1s = np.sum(
        y_test,
    )
    true_n_0s = len(y_test_pred) - true_n_1s
    pred_n_1s = np.sum(y_test_pred)
    pred_n_0s = len(y_test_pred) - pred_n_1s
    comp = [
        (i, y_test[i], y_test_pred[i])
        for i in range(len(y_test))
        if y_test[i] != y_test_pred[i]
    ]
    print(len(y_test))
    print(comp)
    if model < 3:
        print("Training Score: " + str(clf.score(X_train, y_train)))
    print("True n_1s: " + str(true_n_1s))
    print("True n_0s: " + str(true_n_0s))
    print("Pred n_1s: " + str(pred_n_1s))
    print("Pred n_0s: " + str(pred_n_0s))
    print("Loss: " + str(mean_squared_error(y_test, y_test_pred)))
