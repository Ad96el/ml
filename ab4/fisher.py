import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import math
from sklearn.model_selection import train_test_split 
import scipy.stats as stats


def load_from_file(path):
  df = pd.read_csv(path, header=None, sep=",")
  X = df.iloc[:, 0:56].values 
  y = df.iloc[:, 57].values
  return X, y

def splitToClasses (X,y):
    spamx = []
    spamy = []
    mailx = []
    maily = []
    for i in range(len(X)):
        if(y[i] == 1):
            spamx.append(X[i])
            spamy.append(1)
        else:
            mailx.append(X[i])
            maily.append(0)
    return spamx, mailx,spamy, maily

def covariance_matrix(X):
    mu = np.mean(X,axis=0)
    num_samples = len(X)
    X_centered = X - mu # mu is subtracted from all rows of X
    cov = X_centered.transpose().dot(X_centered) / num_samples
    return cov

def fischer(m,s,mm,ms):
    return (   np.linalg.pinv(m+s)@(mm- ms)   )

def buildgauss (x,v):
    project = x@ v
    m_project = np.mean(project)
    var_project= np.var(project)
    return m_project,var_project

def plot(sm,mm,sv,mv):
    sigmaM = math.sqrt(mv)
    sigmaS = math.sqrt(sv)
    x = np.linspace(mm- 3*sigmaM + sv - 3*sigmaS, mm+3*sigmaM + sm+ 3*sigmaS, 50)
    plt.plot(x, stats.norm.pdf(x,mm,sigmaM ))
    plt.plot(x, stats.norm.pdf(x,sm,sigmaS ))
    plt.show()

def gaussCalculate(mean,var,x):
    return 1/(math.sqrt(2* math.pi * var ) ) * math.exp(-1/2 *  ((x-mean)/var )**2  )

def predict(means, variances,test,label):
    result = np.zeros((2, 2))
    for i in range(len(test)):
        gauss1 = gaussCalculate(means[0],variances[0], test[i] )
        gauss0 = gaussCalculate(means[1],variances[1], test[i] )
        if(gauss1 > gauss0):
            result[1][label[i]] +=1
        else:
            result[0][label[i]] +=1
    return result


def main ():
    x,y = load_from_file('spambase.data')
    xtrain , xtest ,  ytrain , ytest = train_test_split(x,y,test_size=0.2)
    spamTrain, mailTrain, yspam , ymail = splitToClasses(xtrain,ytrain)
    covMail = covariance_matrix(mailTrain)
    covSpam = covariance_matrix(spamTrain)
    meanMail = np.mean( mailTrain, axis=0 )
    meanSpam = np.mean(spamTrain, axis=0)
    vec = fischer(covMail, covSpam, meanMail, meanSpam)
    spamMean, spamVariance = buildgauss( spamTrain ,  vec)
    mailMean, mailVariance = buildgauss(mailTrain ,vec )
    plot(spamMean, mailMean ,spamVariance, mailVariance )
    xtest = xtest@ vec
    p = predict( [spamMean,mailMean], [spamVariance,mailVariance], xtest, ytest)
    print(p)



    return

if (__name__ == '__main__'):
    main()