import numpy as np
from numpy.linalg import pinv, eigh
import pandas as pd
from matplotlib import pyplot as plt


training_data = np.array(pd.read_csv('zip.train', sep=' ', header=None))
test_data = np.array(pd.read_csv('zip.test', sep =' ',header=None))
X_train, y_train = training_data[:,1:-1], training_data[:,0]
X_test, y_test = test_data[:,1:], test_data[:,0]

class Classifier:
  
    def accuracy(self, labels, predictions):
        return np.mean(labels == predictions)
    
    def confusion_matrix(self, labels, predictions):
        size = len(set(labels))
        matrix = np.zeros((size, size))
        for correct, predicted in zip(labels.astype(int), predictions.astype(int)):
            matrix[correct][predicted] += 1
        return matrix

class LeastSquares:
    def fit(self, X, y):
        self.w = pinv(X.T@X)@X.T@y 
    def predict(self, X):
        predictions = X@self.w
        return predictions

class OneVsOne(Classifier):
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        classes = list(set(y))
        self.num_classes = len(classes)
        assert classes == list(range(self.num_classes)) # We need the labels to be 0,...,num_classes-1
        self.binary_classifiers = self.build_all_binary_classifiers()
        
    def predict(self, X):
        votes = np.zeros((self.num_classes, len(X)))
        binary_predictions = self.predict_all_binary(X)
        
        for (i,j), prediction in binary_predictions.items():
            votes[i] += (prediction < 0.5).astype(int)
            votes[j] += (prediction >= 0.5).astype(int)
        
        predictions = np.argmax(votes, axis=0)
        return predictions
        
    def predict_all_binary(self, X):
        binary_predictions = {}
 
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
                prediction = self.binary_classifiers[(i,j)].predict(X)
                binary_predictions[(i,j)] = prediction
        return binary_predictions
    
    def build_all_binary_classifiers(self):
        binary_classifiers = {}
        
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
                
                X_train, y_train = self.create_subset(i, j)
                binary_classifiers[(i,j)] = self.build_one_binary_classifier(
                    X_train, y_train)
                
        return binary_classifiers
    
    def build_one_binary_classifier(self, X_train, y_train):
        classifier = self.model()
        classifier.fit(X_train, y_train)
        return classifier
        
    def create_subset(self, i, j):
        X_i = self.X[self.y == i]
        X_j = self.X[self.y == j]
        X = np.concatenate((X_i, X_j))
        y = [0] * len(X_i) + [1] * len(X_j)
        return X, y

 
    
#richtig und fertig
#predict
def predict1( X,w ):
    predictions = X@w
    return predictions

def predict_all_binary( X, a ):
    binary_predictions = {}
    for i in range(10):
        for j in range(i+1, 10):
            asdf = a [(i,j)]
            prediction = predict1(X, asdf )
            binary_predictions[(i,j)] = prediction
    return binary_predictions

def predict( X,a ):
    votes = np.zeros((10, len(X)))
    binary_predictions = predict_all_binary(X,a ) 
    for (i,j), prediction in binary_predictions.items():
        votes[i] += (prediction < 0.5).astype(int)
        votes[j] += (prediction >= 0.5).astype(int)
        
    predictions = np.argmax(votes, axis=0)
    return predictions
#bauen
def build_one_binary_classifier( X, y):
    return pinv(X.T@X)@X.T@y

def create_subset(X,y, i, j):
    X_i = X[y == i]
    X_j = X[y == j]
    X = np.concatenate((X_i, X_j))
    y = [0] * len(X_i) + [1] * len(X_j)
    return X, y

def build_all_binary_classifiers(X,y):
    binary_classifiers = {}    
    for i in range(10):
        for j in range(i+1, 10):       
            X_train, y_train = create_subset(X,y,i, j)
            binary_classifiers[(i,j)] = build_one_binary_classifier(X_train, y_train)
    return binary_classifiers

def fit1( X, y):
    return build_all_binary_classifiers(X,y)
#ende von bauen

def covariance_matrix(X):
    mu = np.mean(X,axis=0)
    num_samples = len(X)
    X_centered = X - mu # mu is subtracted from all rows of X
    cov = X_centered.transpose().dot(X_centered) / num_samples
    return cov

def fit( x, k):
    cov_X = covariance_matrix(x)
    eig_values, eig_vectors = eigh(cov_X)
    return eig_vectors.T[-k:]
        
def project(X, eig_vectors):
    return np.matmul (X,eig_vectors.T )
 





 #validierung
def accuracy(  labels, predictions):
        return np.mean(labels == predictions)

# matrix
def confusion_matrix(  labels, predictions):
        size = len(set(labels))
        matrix = np.zeros((size, size))
        for correct, predicted in zip(labels.astype(int), predictions.astype(int)):
            matrix[correct][predicted] += 1
        return matrix

eig_vectors = fit(X_train, 2)
 
X_train_2d =project(X_train, eig_vectors)
X_test_2d = project(X_test, eig_vectors)

X_train_2d = np.concatenate((np.ones((len(X_train_2d),1)), X_train_2d), axis=1)
X_test_2d = np.concatenate((np.ones((len(X_test_2d),1)), X_test_2d), axis=1)
 
a = fit1(X_train_2d, y_train)
predictions = predict(X_test_2d,a ) 
#fertig
#print(accuracy(y_test, predictions))
#print(confusion_matrix(y_test, predictions))
