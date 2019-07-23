import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 


def load_from_file(path):
  df = pd.read_csv(path, header=None, sep=" ")
  X = df.iloc[:, 1:257].values 
  y = df.iloc[:, 0].values
  return X, y


def plot_digit(X, y, digit):
    X_digit = X[y == digit]
    num_samples = 90
    indices = np.random.choice(range(len(X_digit)), num_samples)
    sample_digits = X_digit[indices]
    fig = plt.figure(figsize=(20, 6))
    for i in range(num_samples):
        ax = plt.subplot(6, 15, i + 1)
        img = sample_digits[i].reshape((16, 16))
        plt.imshow(img, cmap='gray', vmin=-1.0, vmax=1.0)
        plt.axis('off')

    plt.show()


def plot_random_digits(X, y):
    num_samples = 90
    indices = np.random.choice(range(len(y)), num_samples)
    sample_digits = X[indices]
    fig = plt.figure(figsize=(20, 6))
    for i in range(num_samples):
        ax = plt.subplot(6, 15, i + 1)
        img = sample_digits[i].reshape((16, 16))
        plt.imshow(img, cmap='gray', vmin=-1.0, vmax=1.0)
        plt.axis('off')
    plt.show()

def main():
    X_train, y_train = load_from_file("zip.train")
    X_test, y_test = load_from_file("zip.test")
    print('daten eingelesen')
    result =  knn(X_train,X_test,2) 
    print('distanzen berechnet')
    matrix = evaluate(result,y_train, y_test)   # 
    print(matrix)

# calculating the distance between one point of the test set with all points of the training set
def knn (training, test, k):
    results=[]
    for i in range(len(test)):
        distance = []
        for j in range (len (training)):
            dist = np.linalg.norm(training[j] -  test[i])
            distance.append(dist)
        result=[]
        for p in range(k):
            index = distance.index(min(distance))
            result.append(index)
            del distance[index]
        results.append(result)
    return results
#counting the occurrence of a digit in the distance result 
def count (trainLabel, result):
    counter = [0,0,0,0,0,0,0,0,0,0]
    for i in range (len(result)):
        if( int(trainLabel[result[i]]) == 0):
            counter[0] = counter[0] + 1
        if( int(trainLabel[result[i]]) == 1):
            counter[1] = counter[1] + 1
        if( int(trainLabel[result[i]]) == 2):
            counter[2] = counter[2] + 1
        if( int(trainLabel[result[i]]) == 3):
            counter[3] = counter[3] + 1
        if( int(trainLabel[result[i]]) == 4):
            counter[4] = counter[4] + 1
        if( int(trainLabel[result[i]]) == 5):
            counter[5] = counter[5] + 1
        if( int(trainLabel[result[i]]) == 6):
            counter[6] = counter[6] + 1
        if( int(trainLabel[result[i]]) == 7):
            counter[7] = counter[7] + 1
        if( int(trainLabel[result[i]]) == 8):
            counter[8] = counter[8] + 1
        if( int(trainLabel[result[i]]) == 9):
            counter[9] = counter[9] + 1
    return counter.index(max(counter)) 

def evaluate(resultIndex,trainLabel,testLabel):
    matrix =[]
    for i in range (10):
        matrix.append(np.zeros(10))
    for i in range(len(resultIndex)):
        result = count(trainLabel,resultIndex[i])
        a =  matrix[ int (testLabel[i])][result] + 1
        matrix[ int (testLabel[i])] [result] = a
    return matrix

if __name__ == "__main__":
    main()
