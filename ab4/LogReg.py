import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    prob=1-(1/(1+np.exp( vector.T@beta )))
    prob1 = (1/(1+np.exp( vector.T@beta )))
    if prob>= prob1:
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
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
  
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
