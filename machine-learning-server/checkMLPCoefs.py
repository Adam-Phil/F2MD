import os
import joblib

if __name__ == "__main__":
    path = "/home/philip/f2md-training/F2MD/machine-learning-server/saveFile/"
    clfs = [clf for clf in os.listdir(path) if ("MLP" in clf and clf.startswith("clf") and clf.endswith(".pkl") and "atch" in clf)]
    for elem in clfs:
        clf = joblib.load(path+elem)
        with open(path+elem[4:-4], "w") as file:
            file.write("Classes: " + str(clf.classes_) + "\n")
            file.write("Loss: " + str(round(clf.loss_,4)) + "\n")
            file.write("Best Loss: " + str(round(clf.best_loss_,4)) + "\n")
            rounded_losses = [round(loss,4) for loss in clf.loss_curve_]
            file.write("Loss Corve: " + str(rounded_losses) + "\n")
            file.write("Validation Scores: " + str(clf.validation_scores_) + "\n")
            file.write("Best Validation Score: " + str(round(clf.best_validation_score_,4)) + "\n")
            file.write("N Features: " + str(clf.n_features_in_) + "\n")
            file.write("T: " + str(clf.t_) + "\n")
            file.write("N Iterations: " + str(clf.n_iter_) + "\n")
            for i in range(len(clf.coefs_)):
                file.write("----------Coef " + str(i) + ":----------\n")
                coef = clf.coefs_[i]
                file.write(str(coef) + "\n")
        file.close()