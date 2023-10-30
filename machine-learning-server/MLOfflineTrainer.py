if __name__ == "__main__":
    print("Importing...")
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential  
    from keras.layers import Dense, LSTM
    from keras.callbacks import ReduceLROnPlateau
    from keras.utils import to_categorical
    from MLDatasetCollection import determineCurrentCheckVersion, model_name_to_number
    import numpy as np
    import joblib
    import pickle
    import sys
    model = sys.argv[1]
    if model.isdigit():
        model = int(model)
    else:
        model = model_name_to_number(model_name=model)
    partition = sys.argv[3]
    checkVersion = determineCurrentCheckVersion(int(sys.argv[2]))
    extra = "_30_n_iter_100_relu_lbfgs"
    print("Starting...")
    savePath = "/F2MD/machine-learning-server/saveFile"
    if model == 0:
        print("Starting Data Loading; Partition: " + checkVersion + "_" + partition)
        with open (savePath+'/concat_data/'+"valuesSave_SVM_" + checkVersion + "_" + partition + ".listpkl", 'rb') as fp:
            X = pickle.load(fp)
        X = np.array(X)
        with open (savePath+'/concat_data/'+"targetSave_SVM_" + checkVersion + "_" + partition + ".listpkl", 'rb') as ft:
            y = pickle.load(ft)
        y = np.array(y)
        print("Data loaded")
        clf = SVC(gamma=0.001, C=100., verbose=1)
        print("Starting machine learning: SVM")
        clf.fit(X, y)
        print("Machine Learning done; saving model")
        joblib.dump(clf, savePath + '/clfs/clf_SVM_SINGLE_' + checkVersion + "_" + partition + ".pkl")
    elif model == 1:
        print("Starting Data Loading; Partition: " + checkVersion + "_" + partition)
        with open (savePath+'/concat_data/'+"valuesSave_MLP_L1N25_" + checkVersion + "_" + partition + ".listpkl", 'rb') as fp:
            X = pickle.load(fp)
        X = np.array(X)
        with open (savePath+'/concat_data/'+"targetSave_MLP_L1N25_" + checkVersion + "_" + partition + ".listpkl", 'rb') as ft:
            y = pickle.load(ft)
        y = np.array(y)
        print("Data loaded")
        clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(25,))
        print("Starting machine learning: MLP_L1N25")
        clf.fit(X, y)
        print("Machine Learning done; saving model")
        joblib.dump(clf, savePath + '/clfs/clf_MLP_SINGLE_L1N25_' + checkVersion + "_" + partition + ".pkl")
    elif model == 2:
        print("Starting Data Loading; Partition: " + checkVersion + "_" + partition)
        with open (savePath+'/concat_data/'+"valuesSave_MLP_L3N25_" + checkVersion + "_" + partition + ".listpkl", 'rb') as fp:
            X = pickle.load(fp)
        X = np.array(X)
        with open (savePath+'/concat_data/'+"targetSave_MLP_L3N25_" + checkVersion + "_" + partition + ".listpkl", 'rb') as ft:
            y = pickle.load(ft)
        y = np.array(y)
        print("Data loaded")
        clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(25,25,25), solver="lbfgs", activation="relu", random_state=1, n_iter_no_change=100, max_iter=100000)
        print("Starting machine learning: MLP_L3N25")
        clf.fit(X, y)
        print("Machine Learning done; saving model")
        joblib.dump(clf, savePath + '/clfs/clf_MLP_SINGLE_L3N25_' + checkVersion + "_" + partition + ".pkl")
    elif model == 3:
        print("Starting Data Loading; Partition: " + checkVersion + "_" + partition)
        with open (savePath+'/concat_data/'+"valuesSave_LSTM_" + checkVersion + "_" + partition + ".listpkl", 'rb') as fp:
            X = pickle.load(fp)
        X = np.array(X)
        with open (savePath+'/concat_data/'+"targetSave_LSTM_" + checkVersion + "_" + partition + ".listpkl", 'rb') as ft:
            y = pickle.load(ft)
        y = np.array(y)
        y = to_categorical(y)
        print(X.shape)
        print(y.shape)
        print("Data loaded")
        clf = Sequential()
        clf.add(LSTM(128, return_sequences=True, input_shape=(len(X[0]), len(X[0][0]))))
        clf.add(LSTM(128, return_sequences=True))
        clf.add(LSTM(128, return_sequences=False))
        clf.add(Dense(y.shape[1],activation='softmax'))  
        clf.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.7,patience=4, min_lr=0.0005)
        clf.fit(X, y,epochs=10, batch_size=64, callbacks=[reduce_lr])
        print("Starting machine learning: LSTM")
        clf.fit(X, y)
        print("Machine Learning done; saving model")
        joblib.dump(clf, savePath + '/clfs/clf_LSTM_RECURRENT_' + checkVersion + "_" + partition + ".pkl")
    else:
        raise ValueError("Not a suitable model selected")
    if model == 2 or model == 3:
        print("Classes: " + str(clf.classes_) + "\n")
        print("Loss: " + str(round(clf.loss_,4)) + "\n")
        for i in range(len(clf.coefs_)):
            print("----------Coef " + str(i) + ":----------\n")
            coef = clf.coefs_[i]
            print(str(coef) + "\n")
        # print("Best Loss: " + str(round(clf.best_loss_,4)) + "\n")
        # rounded_losses = [round(loss,4) for loss in clf.loss_curve_]
        # print("Loss Corve: " + str(rounded_losses) + "\n")
        # print("Validation Scores: " + str(clf.validation_scores_) + "\n")
        # print("Best Validation Score: " + str(round(clf.best_validation_score_,4)) + "\n")
        print("N Features: " + str(clf.n_features_in_) + "\n")
        print("T: " + str(clf.t_) + "\n")
        print("N Iterations: " + str(clf.n_iter_) + "\n")
        
