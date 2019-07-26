import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def load_from_fileTest(path):
  df = pd.read_csv(path, header=None, sep=" ")
  X = df.iloc[:, 1:257].values 
  y = df.iloc[:, 0].values
  return X, y

def load_from_file(path):
  df = pd.read_csv(path, header=None, sep=",")
  X = df.iloc[:].values 
  return X

def gauss_classifier (sigma,avg,x):
  avg = np.matrix(avg)
  sigma=np.matrix(sigma)
  return (np.log(np.power((2*np.pi),-len(avg)/2) * np.power(np.linalg.det(sigma),-0.5) )) - (0.5*np.linalg.det((x-avg)*np.linalg.inv(sigma)*np.transpose(x-avg)))    

def avg(train,k):
    bound = np.zeros(k)
    for i in train:
        bound = np.add(bound, i)
    return np.multiply(bound,1/len(train))

def cov(train, avg):
    res = np.zeros((256,256))
    for i in train:
        norm = np.subtract(i, avg).reshape(1,-1)
        transported = norm.reshape(-1,1)
        prod = np.matmul(transported, norm)
        res+=prod
    I=np.eye(len(avg))
    res /= len(train)
    while np.linalg.det(res)==0:
      res=(0.001*I+0.999*res)
    return res

def classifier(sigmas,avgs,test,label):
  matrix = np.zeros((10,10))
  for i in range(len(test)):
    probs = []
    for j in range(10):
      prob = gauss_classifier(sigmas[j], avgs[j],test[i])
      probs.append(prob)
    res = probs.index(max(probs))
    matrix[int(label[i])] [res] +=1 
  print(matrix)

def main():
  sigmas = []
  avgVec = []
  k = 256
  X_test, y = load_from_fileTest("zip.test")
  for i in range(10):
    file = str(i) + ".train"
    X_train = load_from_file(file)
    avgVector = avg(X_train,k)
    coVariance = cov(X_train,avgVector)
    sigmas.append(coVariance)
    avgVec.append(avgVector)
  classifier(sigmas,avgVec,X_test,y)
  return 

if __name__ == "__main__":
  main()