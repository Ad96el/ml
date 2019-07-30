import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
def covariance_matrix(X, mu):
    num_samples = len(X)
    X_centered = X - mu # mu is subtracted from all rows of X
    cov = X_centered.transpose().dot(X_centered) / num_samples
    return cov


def gauss(x_train, y_train):
  labels = np.unique(y_train)
  means = []
  covariances = []
  
  for label in labels:
    X_sub = x_train[y_train == label]
    mean = np.mean(X_sub, axis=0)
    means.append(mean)
    cov = covariance_matrix(X_sub, mean)
    I=np.eye(len(cov[0]))
    while np.linalg.det(cov)==0:
      cov=(0.001*I+0.999*cov)
    covariances.append(cov)
  
  return labels, means, covariances


def log_normal_distribution_pdf(X, sigma, mu):
  dim = X.shape[1]
  log_normalization_term = -dim / 2 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(sigma))

  X_centered = X - mu
  exponent = -0.5 * np.sum((X_centered).transpose() * (np.linalg.inv(sigma).dot((X_centered).transpose())), axis=0)

  return log_normalization_term + exponent


def predict(labels, means, covariances, X):
  # vectorized prediction
  results = np.zeros(len(X), dtype=int)
  largest_log_probs = np.ones(len(X)) * -np.inf  # vector to save the largest log_probability for each sample
  # iterate over all gaussian classifiers / classes
  for label, mean, covariance in zip(labels, means, covariances):
    log_probs = log_normal_distribution_pdf(X, covariance, mean)  # log(p(X)) for all data points -> vector of len(X) log_probabilities
    results[log_probs > largest_log_probs] = label  # update class label where a new maximum was found
    largest_log_probs = np.maximum(largest_log_probs, log_probs)  # update largest log probabilities

  return results
'''

def confusion_matrix(y_true, y_predicted):
  size = len(set(y_true))
  results = np.zeros((size, size), dtype=np.int32)

  for yi, pi in zip(y_true, y_predicted):
      results[int(yi)][int(pi)] += 1

  return results


def predict_LogReg(beta,x):
  results=np.zeros(len(x))
  i=0
  for vector in x:
    prob=1-(1/(1+np.exp(vector.dot(np.transpose(beta)))))
    if prob>=0.5:
      results[i]=1
    i+=1

  return results


def gradient(beta,x_,y_):
  grad=np.zeros(len(x_[0]))
  i=0
  for x in x_:
    prob=1-(1/(1+np.exp(x.dot(np.transpose(beta)))))
    grad+=x*(y_[i]-prob)
    i+=1

  return grad

def normalize(x):
  #Zentrierung
  mean = np.mean(x, axis=0)
  x=x-mean

  #Normalisierung der Längen: Nicht mit der Varianz, sondern so, dass jede Komponente maximal 1 ist
  maximums=np.zeros(len(x[0]))
  for vector in x:
    i=0
    for e in vector:
      if e>maximums[i]:
        maximums[i]=e
      i+=1

  x=x/maximums
  
  return x

def main():
  data = pd.read_csv("spambase.data", header=None).values
  x = data[:, :-1]
  y = data[:, -1]
  x=normalize(x)
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=30,stratify=y)
  
  #labels, means, covariances = gauss(x_train, y_train)
  #y_pred = predict(labels, means, covariances, x_test)

  alpha=0.1
  iterations=100
  beta=np.zeros(len(x_test[0]))
  scores=[]
  x_grid=np.arange(iterations)
  i=0
  while i<iterations:
    y_pred = predict_LogReg(beta, x_test)
    score = np.mean(y_pred == y_test)
    scores.append(score)
    beta+=alpha*gradient(beta,x_train,y_train)
    i+=1

  plt.plot(x_grid, scores)
  plt.xlabel("Iterationen")
  plt.ylabel("score")
  plt.title("alpha: "+ str(alpha))
  plt.show()
  
  print("final score: {}".format(score))

  KM = confusion_matrix(y_test, y_pred)
  print("final confusion matrix: (rows=y_true columns=y_predicted)")
  for e in KM:
    print(e)
    
    

if __name__ == "__main__":
  main()
