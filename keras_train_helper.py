from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import datetime
import re
import shutil
import keras.backend as K
import tensorflow as tf

def tfauc(y_true, y_pred):
     auc = tf.metrics.auc(y_true, y_pred)[1]
     K.get_session().run(tf.local_variables_initializer())
     return auc

def rotateTensorboardLogs():
    copyEnding = re.sub(r"[\s\:\.]", "-", str(datetime.datetime.now()))
    shutil.move("./tb-logs", "./tb-logs-{}".format(copyEnding))

class PrintAucCallback(Callback):
    def __init__(self, sentences, labels, printPerBatches=None):
        self.testData = [sentences, labels]
        self.smallTestData = [sentences[:int(sentences.shape[0] / 10)], labels[:int(labels.shape[0] / 10)]]
        self.printPerBatches = printPerBatches
        self.listOfAucs = []

    def computeAuc(self, small=False):
        sentences, labels = self.smallTestData if small else self.testData
        predictions = self.model.predict(sentences)
        return roc_auc_score(labels[:, 1], predictions[:, 1])
        
    def printAuc(self, auc):
        print(" - auc: {:.4f}".format(auc))
        
    def on_batch_end(self, batch, logs={}):
        if self.printPerBatches and (batch + 1) % self.printPerBatches == 0:
            self.printAuc(self.computeAuc())
    
    def on_epoch_end(self, epoch, logs={}):
        auc = self.computeAuc() if (epoch + 1) % 5 == 0 else self.computeAuc(small=True)
        self.printAuc(auc)
        self.listOfAucs.append((epoch + 1, auc))
        