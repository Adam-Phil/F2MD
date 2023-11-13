from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PartialVisualTrainer:
    def __init__(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.clf = MLPClassifier(
            alpha=1e-5,
            hidden_layer_sizes=(25,),
            verbose=True,
            warm_start=True,
            solver='lbfgs',
            max_iter=10,
            n_iter_no_change=9,
        )
        self.coef_store = []

    def animate(self, i):
        self.train_MLP()
        weights = self.coef_store[i][0]
        ax.cla()
        sns.heatmap(ax=ax, data=weights, annot=True, fmt=".3f", cbar_ax=cbar_ax)

    def train_MLP(self):
        self.clf.fit(self.X_train, self.y_train)
        weights = self.clf.coefs_[0]
        biases = self.clf.coefs_[1]
        self.coef_store.append((weights, biases))
        print("Loss: " + str(round(self.clf.loss_, 4)) + "\n")


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
    trainer = PartialVisualTrainer(X_train, y_train)

    grid_kws = {"width_ratios": (0.9, 0.05), "wspace": 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(20, 16))
    ani = FuncAnimation(fig=fig, func=trainer.animate, frames=100, interval=100)
    plt.show()
