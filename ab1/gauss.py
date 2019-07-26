import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_from_file(path):
  df = pd.read_csv(path, header=None, sep=" ")
  X = df.iloc[:, 1:257].values # empty string at position 257: every line ends with a space (== separator)
  y = df.iloc[:, 0].values
  return X, y


def get_ny(vectors):
  ny=np.zeros(len(vectors[0]))

  for vector in vectors:
    i=0
    for e in vector:
      ny[i]+=e
      i+=1
  i=0
  while i<len(ny):
    ny[i]=ny[i]/len(vectors)
    i+=1
  return ny
    

def get_sigma(vectors,ny_in):
  sigma=np.zeros(shape=(len(vectors[0]),len(vectors[0])))
  ny=np.matrix(ny_in)
  
  for vector in vectors:
    v=np.matrix(vector)
    sigma+=np.transpose(v-ny)*(v-ny)
  
  sigma=sigma/len(vectors)
  
  return sigma


def gauss(x_train, y_train):
  zahlen=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
  i=0
  for e in y_train:
    zahlen[int(e)].append(x_train[i])
    i+=1

  i=0
  nys=[]
  for e in zahlen:
    del e[0]
    nys.append(get_ny(zahlen[i]))
    i+=1
  i=0
  sigmas=[]
  for e in zahlen:
    sigmas.append(get_sigma(e,nys[i]))
    i+=1

  return nys, sigmas


def prob(vector_in, ny_in, sigma_in):
  vector=np.matrix(vector_in)
  ny=np.matrix(ny_in)
  sigma=np.matrix(sigma_in)
  I=np.eye(len(vector_in))
  
  while np.linalg.det(sigma)==0:
    sigma=(0.001*I+0.999*sigma)
  
  return (np.log(np.power((2*np.pi),-len(vector)/2) * np.power(np.linalg.det(sigma),-0.5) )) - (0.5*np.linalg.det((vector-ny)*np.linalg.inv(sigma)*np.transpose(vector-ny)))


def classifier(nys, sigmas, points, values):
  KM = np.zeros((10,10)) # Konfusionsmatrix 10x10
  i=0
  for point in points:
    probabilities=[]
    for z in range(10):
      probabilities.append(prob(point,nys[z],sigmas[z]))
      z+=1
    KM[int(values[i])][probabilities.index(max(probabilities))]+=1
    if i%100==0:
      print(i)
    i+=1
  return KM


def main():
  x_train, y_train = load_from_file("zip.train")
  x_test, y_test = load_from_file("zip.test")
  
  nys, sigmas = gauss(x_train, y_train)
  print(nys)
  #print(sigmas)
  
  KM=classifier(nys, sigmas, x_test, y_test)
  
  for e in KM:
    print(e)
    
    

if __name__ == "__main__":
  main()
