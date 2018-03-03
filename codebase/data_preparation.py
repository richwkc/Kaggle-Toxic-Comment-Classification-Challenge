import pandas as pd
import numpy as np
from os import path
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.keyedvectors import KeyedVectors


def loadData():
    print("Loading datasets...")
    importDirectory = "../../state/data/preprocessed-train-test/"

    train, test, allData, contestTest = map(
        lambda filename: pd.read_csv(path.join(importDirectory, filename)), 
        ["train.csv", "test.csv", "all.csv", "contest-test.csv"])

    print("train: {}, test: {}, allData: {}, contestTest: {}".format(
        train.shape, test.shape, allData.shape, contestTest.shape))
    
    return train, test, allData, contestTest


class SentenceSplitter(BaseEstimator, TransformerMixin): 
    def __init__(self, column):
        self.column = column
    
    def fit(self):
        return self
    
    def transform(self, X):
        print("Splitting sentences...")        
        return (X[self.column]
            .str.replace("[^A-Za-z\s]", "")
            .str.lower()
            .str.split())

    
class Word2Int(BaseEstimator, TransformerMixin):
    def __init__(self, completeDataset):
        print("Loading w2i and i2w dictionaries...")
        self.allWords = set([word for sentence in completeDataset for word in sentence])

        self.w2i = { word: index for index, word in enumerate(self.allWords) }
        self.i2w = { index: word for index, word in enumerate(self.allWords) }
        
    def fit(self):
        return self    
    
    def transform(self, X):
        print("Converting words to integers...")        
        return X.apply(lambda sentence: [self.w2i[word] for word in sentence])

    
class Word2Vec:
    def __init__(self, filepath, dimensions, i2w, seed=None):
        self.dictionary = set([word for word in i2w.values()])
        self.i2w = i2w
        self.dimensions = dimensions
        if seed:
            np.random.seed(seed)
        
        print("Loading word2vec dictionary...")
        self.embedding = KeyedVectors.load(filepath, mmap="r")
    
    def embeddingMatrix(self):
        availableWords = set.intersection(self.dictionary, set(self.embedding.vocab.keys()))
        
        assert len(availableWords) > 100000
        
        int2vec = {index: self.embedding.word_vec(word) 
            if word in availableWords 
                else np.random.normal(scale=.644, size=(self.dimensions,))
            for index, word in self.i2w.items()}
        
        matrix = np.array([vector for vector in int2vec.values()])
        
        return matrix, int2vec
    
class Oversampler(BaseEstimator, TransformerMixin):
    def __init__(self, label):
        self.label = label
    
    def fit(self):
        return self
    
    def transform(self, X):
        print("Oversampling...")
        multiples = int(X[X[self.label] == 0].shape[0] / X[X[self.label] == 1].shape[0])

        datasetPositive = X[X[self.label] == 1]

        return pd.concat([X] + multiples * [datasetPositive]).reset_index()
    
class ZeroPadder(BaseEstimator, TransformerMixin):
    def __init__(self, maxSeqLength):
        self.maxSeqLength = maxSeqLength
    
    def __padArray__(self, array):
        fullArray = np.zeros(self.maxSeqLength)
        fullArray[:min(array.shape[0], self.maxSeqLength)] = array[:min(array.shape[0], self.maxSeqLength)]
        return fullArray
    
    def fit(self):
        return self
    
    def transform(self, X):
        print("Zero-padding...")
        return np.array(X
            .apply(lambda l: self.__padArray__(np.array(l)))
            .tolist())
    
class Labelizer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self):
        return self
    
    def transform(self, X):
        return np.array(X[self.column]
            .apply(lambda label: np.array([0, 1]) if label == 1 else np.array([1, 0]))
            .tolist())