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

def predict(w1,w2,w3,test,label):
    print(w1)
    for i in  range(20):
            #print(test[i])
            o0_ = cont(test[i])
            o1 = sig(o0_@w1)
            o1_ = cont(o1)
            o2 = sig(o1_@w2)
            o2_ = cont(o2)
            o3 = sig(o2_@w3)
            #print(o3)
            print(str(o3.index(max(o3))) + " " + str(label[i]) )
    #print('______________')

def sig(vec,bol=True ):
    result = []
    for i in vec:
        result.append( 1/(1+np.exp(-i )))
    return result

def cont(vec):
    return np.concatenate((vec,[1]))


def error(vec,val):
    t = np.zeros(len(vec))
    t[val] = 1
    return vec - t 

def asdf(vec):
    result = np.zeros(  (len(vec),len(vec)) )
    for i in range(len(vec)):
        result[i][i] += vec[i]*(1-vec[i])
    return result

def fit(x,y,test,label):
    omega = -0.1
    w1 = np.random.rand(257,40)#*0.08-0.04
    w2 = np.random.rand(41,50)#*0.08-0.04
    w3 = np.random.rand(51,10)#*0.08-0.04
    deltaW2_ = 0
    deltaW3_ = 0
    deltaw1_ = 0
    for j in range(20):
        #deltaW2_ = 0
        #deltaW3_ = 0
        #deltaw1_ = 0
        for i in  range(len(x) ):
            #ff 
            o0_ = cont(x[i])
            o1 = sig(o0_@w1)
            o1_ = cont(o1)
            o2 = sig(o1_@w2)
            o2_ = cont(o2)
            o3 = sig(o2_@w3)
            #bp 
            e = error(o3, int(y[i]) )
            D1 = asdf(o1)
            D2 = asdf(o2)
            D3 = asdf(o3)
            d3 = D3@e
            w3_ = np.delete(w3, len(w3)-1,0 )
            d2 = D2@w3_@d3
            w2_ = np.delete(w2, len(w2)-1,0 )
            d1 = D1@w2_@d2
            o2_ = o2_.reshape(-1,1)
            o1_ = o1_.reshape(-1,1)
            o0_ = o0_.reshape(-1,1)
            deltaW3_ += (omega * d3.reshape(-1,1) @ o2_.T ).T
            deltaW2_ += (omega * d2.reshape(-1,1) @ o1_.T ).T
            deltaw1_ += (omega * d1.reshape(-1,1) @ o0_.T ).T
        w1 = deltaw1_
        w2 = deltaW2_
        w3 = deltaW3_
        predict(w1,w2,w3,test,label)
    #return w1,w2,w3

def main():
    xtrain, y = load_from_file('zip.train')
    xtest,ytest = load_from_file('zip.test')
    fit(xtrain,y,xtest,ytest )
    return




if __name__ == '__main__':
    main()