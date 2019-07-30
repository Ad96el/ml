import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

 
 
def load_from_file(path):
  df = pd.read_csv(path, header=None, sep=" ")
  X = df.iloc[:, 1:257].values 
  y = df.iloc[:, 0].values
  return X, y
 

def loadImage():
  imgs = []
  for filename in os.listdir("faces"):
    img =[]
    for i in plt.imread("faces/" + filename):
      img.extend(i)
    imgs.append(img)
  return np.array(imgs) 
 
def plot(train, y):
        plt.figure(figsize=(100, 50))
        pos = 0
        for i in range(10):
            for j in range(i+1, 10):
                pos += 1
                plt.subplot(9, 5, pos)
                plot_classes(train, y, i, j)
        plt.show()

def plot_classes(project, label,i, j ):
  class_i_x = []
  class_i_y = []
  class_j_x = []
  class_j_y = []
  for index in range(len(label)):
    if(i == label[index]):
      class_i_x.append(project[index][0] )
      class_i_y.append(project[index][1] )
    if(j == label[index]):
      class_j_x.append(project[index][0])
      class_j_y.append(project[index][1] )
  plt.scatter(class_i_x, class_i_y)
  plt.scatter(class_j_x, class_j_y)
 
 

def plot_faces(vec):
  images = vec.reshape(20, 64,64)
  plt.figure(figsize=(150,150))
  for i in range( len(images)):
    plt.subplot(9,5,i+1)
    plt.imshow( images[i] )
  plt.show()

 
def classifiy( X, betas):
    result = {}
    for i in range(10):
        for j in range(i+1, 10):
            beta = betas[(i,j)]
            res = np.matmul(X,beta)
            result[(i,j)] = res
    return result


def predict( X,betas ):
    counter = np.zeros((10, len(X)))
    predij = classifiy(X,betas )
    for (i,j), p in predij.items():
      result_i = (p < 0).astype(int)
      counter [i] += result_i
      result_j = (p>= 0).astype(int)
      counter [j] += result_j     
    return np.argmax(counter, axis=0)
 

def classij(X,y, i, j):
    xi= []
    xj= []
    for index in range(len(y)):
      if(y[index] == i):
        xi.append(X[index])
      if(y[index] == j):
        xj.append(X[index])
    xij = np.concatenate((xi, xj))
    y = np.concatenate( (np.ones(len(xi)) * -1  ,np.ones( len(xj))) ) 
    return xij, y

def calculate_beta(X,y):
    betas = {}    
    for i in range(10):
        for j in range(i+1, 10):       
            x, yt = classij(X,y,i, j)
            betas[(i,j)] = np.matmul( np.linalg.pinv( np.matmul (x.T,x ) ) , np.matmul(x.T, yt))
    return betas



def covariance_matrix(X):
    mu = np.mean(X,axis=0)
    num_samples = len(X)
    X_centered = X - mu # mu is subtracted from all rows of X
    cov = X_centered.transpose().dot(X_centered) / num_samples
    return cov

def vectors( x, k):
    cov = covariance_matrix(x)
    val, vec = np.linalg.eigh(cov)
    return vec.T[-k:]
        
 

# matrix
def confusion_matrix( y_true, y_predicted):
        size = len(set(y_true))
        results = np.zeros((size, size), dtype=np.int32)
        for yi, pi in zip(y_true, y_predicted):
            results[int(yi)][int(pi)] += 1
        return results
 
def main():
    X_train, y_train = load_from_file('zip.train')
    X_test, y_test = load_from_file('zip.test')
    img = loadImage()
 
    vec = vectors(X_train, 2)
 
    X_train_2d =np.matmul(X_train,vec.T )
    X_test_2d = np.matmul(X_test,vec.T )

    #plot digits
    plot(X_train_2d,y_train  )

    X_train_2d = np.concatenate((np.ones((len(X_train_2d),1)), X_train_2d), axis=1)
    X_test_2d = np.concatenate((np.ones((len(X_test_2d),1)), X_test_2d), axis=1)
 
    betas = calculate_beta(X_train_2d, y_train)
    result = predict(X_test_2d, betas )
    print(confusion_matrix(y_test, result))
   
    #faces
    dim_faces = 20
    cov_face = covariance_matrix(img)
    vec_face = vectors(cov_face, dim_faces ) 
    plot_faces(vec_face)
    return
 
 
if __name__ == "__main__":
  main()