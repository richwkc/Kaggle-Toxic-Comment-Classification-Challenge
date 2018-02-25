from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import datetime
import re
import shutil
import keras.backend as K
import tensorflow as tf
import numpy as np
import os.path 

def tfauc(y_true, y_pred):
     auc = tf.metrics.auc(y_true, y_pred)[1]
     K.get_session().run(tf.local_variables_initializer())
     return auc

def rotateTensorboardLogs():
    if os.path.isdir("./tb-logs"):
        copyEnding = re.sub(r"[\s\:\.]", "-", str(datetime.datetime.now()))
        shutil.move("./tb-logs", "./tb-logs-{}".format(copyEnding))

class PrintAucCallback(Callback):
    def __init__(self, sentences, labels, printPerBatches=None):
        np.random.seed(64563)
        tenthOfCount = int(sentences.shape[0] / 10)
        
        self.testData = [sentences, labels]
        self.smallTestData = [
            sentences[np.random.choice(sentences.shape[0], tenthOfCount, replace=False)], 
            labels[np.random.choice(labels.shape[0], tenthOfCount, replace=False)] ]
        self.printPerBatches = printPerBatches
        self.listOfAucs = []

    def computeAuc(self, smallTestSet=False):
        sentences, labels = self.smallTestData if smallTestSet else self.testData
        predictions = self.model.predict(sentences)
        return roc_auc_score(labels[:, 1], predictions[:, 1])
        
    def printAuc(self, auc, smallTestSet=False):
        largeTestSet = "" if smallTestSet else "all-test-data-"
        print(" - {}auc: {:.4f}".format(largeTestSet, auc))
        
    def on_batch_end(self, batch, logs={}):
        if self.printPerBatches and (batch + 1) % self.printPerBatches == 0:
            self.printAuc(self.computeAuc())
    
    def on_epoch_end(self, epoch, logs={}):
        smallTestSet = (epoch + 1) % 5 != 0
        auc = self.computeAuc(smallTestSet=smallTestSet)
        
        self.printAuc(auc, smallTestSet=smallTestSet)
        self.listOfAucs.append((epoch + 1, auc))
        