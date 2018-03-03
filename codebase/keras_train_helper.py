from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import datetime
import re
import shutil
import keras.backend as K
import tensorflow as tf
import numpy as np
import os.path
import math
import matplotlib.pyplot as plt


def getEpochIndices(listOfAucs):
    return [index 
            for index, (iterationType, number, value) 
            in zip(range(len(listOfAucs)), listOfAucs)
            if iterationType == "epoch"]


def plotLearningCurve(listAucsTrain, listAucsTest):
    plt.plot([value for _, key, value in listAucsTrain])
    plt.plot([value for _, key, value in listAucsTest])

    for index in getEpochIndices(listAucsTrain):
        plt.axvline(index, ls="--", color="olive")

    plt.legend(["train", "test"])
    plt.ylabel("Area under ROC")
    plt.xlabel("Epochs")
    plt.show()

def tfauc(y_true, y_pred):
     auc = tf.metrics.auc(y_true, y_pred)[1]
     K.get_session().run(tf.local_variables_initializer())
     return auc


def rotateTensorboardLogs():
    if os.path.isdir("./tb-logs"):
        copyEnding = re.sub(r"[\s\:\.]", "-", str(datetime.datetime.now()))
        shutil.move("./tb-logs", "./tb-logs-{}".format(copyEnding))


def printAuc(auc, datasetName):
    print(" - {} auc: {:.4f}".format(datasetName, auc))


class PrintAucCallback(Callback):
    def __init__(self, batchSize, printFrequency=None):
        np.random.seed(64563)
        
        self.batchSize = batchSize
        self.printPerBatches = math.ceil(1 / printFrequency)
        self.listOfAucsTrain = []
        self.listOfAucsTest = []

    def setDatasets(self, train, test):
        samples = np.random.choice(train[0].shape[0], int(train[0].shape[0] / 5), replace=False)
        self.smallTrain = [train[0][samples], train[1][samples]]
        self.train = train
        self.test = test
        
    def __computeAuc__(self, dataset):
        sentences, labels = dataset
        predictions = self.model.predict(sentences, batch_size=self.batchSize)
        return roc_auc_score(labels[:, 1], predictions[:, 1])

    def __handleAuc__(self, iterationType, count, subsampleTrain=False):
        trainAuc = self.__computeAuc__(self.smallTrain if subsampleTrain else self.train)
        testAuc = self.__computeAuc__(self.test)

        printAuc(trainAuc, "train")
        printAuc(testAuc, "test")

        self.listOfAucsTrain.append((iterationType, count, trainAuc))
        self.listOfAucsTest.append((iterationType, count, testAuc))

    def on_batch_end(self, batch, logs={}):
        if self.printPerBatches and (batch + 1) % self.printPerBatches == 0:
            self.__handleAuc__("batch", batch + 1, subsampleTrain=True)

    def on_epoch_end(self, epoch, logs={}):
        self.__handleAuc__("epoch", epoch + 1, subsampleTrain=True)
        plotLearningCurve(self.listOfAucsTrain, self.listOfAucsTest)
