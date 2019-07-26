import numpy as np
import pandas as pd
from numpy.linalg import pinv, eigh
from matplotlib import pyplot as plt
import os



class PCA:   
    def fit(self, X, k):
        cov_X = np.cov(X, rowvar=False)
        eig_values, eig_vectors = eigh(cov_X)
        return eig_vectors.T[-k:]
    
    def project(self, X, eig_vectors):
        return X@eig_vectors.T

pca = PCA()
