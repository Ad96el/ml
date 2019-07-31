import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import math
from sklearn.model_selection import train_test_split 
import scipy.stats as stats

scores = []

def load_from_file(path):
  df = pd.read_csv(path, header=None, sep=" ")
  X = df.iloc[:, 1:257].values 
  y = df.iloc[:, 0].values
  return X, y

def sig(vec,bol=True ):
    result = []
    for i in vec:
        result.append( 1/(1+np.exp(-i )))
    return result

def cont(vec):
    return np.concatenate((vec,[1]))


def error(vec):
    t = np.ones(len(vec))
    return vec - t   

def asdf(vec):
    result = np.zeros(  (len(vec),len(vec)) )
    for i in range(len(vec)):
        result[i][i] += vec[i]*(1-vec[i])
    return result

def fit(x):
    w0 = np.random.rand(257,40)
    w1 = np.random.rand(41,50)
    w2 = np.random.rand(51,10)
    for j in range(1):
        for i in x:
            #ff
            o0 = np.concatenate((i,[1]))
            o1 = sig(o0@w0)
            o1_ = cont(o1)
            o2 = sig(o1_@w1)
            o2_ = cont(o2)
            o3 = sig(o2_@w2)
            #bp 
            e = error(o3)
            D1 = asdf(o1)
            D2 = asdf(o2)
            D3 = asdf(o3)
            d3 = D3@e
            print(d3)
            w2_ = np.delete(w2, len(w2)-1 )
            d2 = D2@w2_ #@d3
            w1_ = np.delete(w1, len(w1)-1 )
            d1 = D1@w1@d2
 
    return

def main():
    xtrain, y = load_from_file('zip.train')
    xtest,ytest = load_from_file('zip.test')
    fit(xtrain)
    return




if __name__ == '__main__':
    main()