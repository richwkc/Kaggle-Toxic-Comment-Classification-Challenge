from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score
import numpy as np
from IPython.display import HTML, display
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(precision=4, suppress=True)


def getMetrics(labels, predictions, probPredictions):
    return {"Confusion Matrix": pd.DataFrame(
        data=confusion_matrix(labels, predictions),
        index=["T Neutral", "T Toxic"],
        columns=["P Neutral", "P Toxic"]),
        "Relativized Confusion Matrix": pd.DataFrame(
            data=confusion_matrix(labels, predictions) / float(len(predictions)),
            index=["T Neutral", "T Toxic"],
            columns=["P Neutral", "P Toxic"]),
        "F1 score": round(f1_score(labels, predictions, pos_label=1.0), 3),
        "Logarithmic loss": round(log_loss(labels, probPredictions), 4),
        "Area under ROC": roc_auc_score(labels, probPredictions[:, 1])}


def printMetrics(metrics):
    for (name, value) in metrics.items():
        display(HTML("<div style='font-weight:bold'>{} :</div>".format(name)))
        if name == "Area under ROC":
            display(HTML("<div style='font-weight:bold; color:red'>{}</div>".format(value)))
        else:
            print(value)


def showROC(labels, probPredictions):
    fpr, tpr, _ = roc_curve(labels, probPredictions, pos_label=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.plot([0, 1], [0, 1], color="darkorange", linestyle="--")
    plt.plot(fpr, tpr, color="steelblue")
    plt.show()


def evaluatePredictions(labels, predictions, probPredictions):
    printMetrics(getMetrics(labels, predictions, probPredictions))
    showROC(labels.as_matrix(), probPredictions[:, 1])
