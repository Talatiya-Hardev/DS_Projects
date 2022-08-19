# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:15:00 2022

@author: Hardev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
#%%
mat = loadmat('ex8data1.mat')
X = mat['X']
Xval = mat['Xval']
yval = mat['yval']
plt.scatter(X[:, 0], X[:, 1], marker='x', alpha=0.5)
plt.xlim(0,30)
plt.ylim(0,30)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
#%%
def estimateGaussian(X):
    m, n = X.shape
    mu = (1/m) * np.sum(X, axis=0)
    sigma2 = (1/m) * np.sum((X - mu)**2, axis=0)
    return mu, sigma2

#%%
def multivariateGaussian(X, mu, sigma2):
    k = len(mu)
    sigma2 = np.diag(sigma2)
    X = X - mu.T
    p = 1/ ((2*np.pi)**(k/2) * (np.linalg.det(sigma2)**0.5)) * np.exp( -0.5 * np.sum(X @ np.linalg.pinv(sigma2) * X, axis=1))
    return p
mu, sigma2 = estimateGaussian(X)
p = multivariateGaussian(X, mu, sigma2)
# Visualize Distribution
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1],marker="x")
X1, X2 = np.meshgrid(np.linspace(0, 35, num=71), np.linspace(0, 35, num=71))
p2 = multivariateGaussian(np.column_stack((X1.flatten().reshape(-1, 1), X2.flatten().reshape(-1, 1))), mu, sigma2)
contour_level = np.power(10, np.array([np.arange(-20, 0, 3, dtype=np.float)]))[0]
plt.contour(X1, X2, p2.reshape(X1.shape), contour_level)
plt.xlim(0,35)
plt.ylim(0,35)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
#%%
def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    step = (max(pval) - min(pval))/1000
    epi_range = np.arange(min(pval), max(pval), step)
    for epsilon in epi_range:
        
        predictions = (pval < epsilon).reshape(-1, 1)
        
        true_positive = np.sum(predictions[yval==1]==1)
        false_positive = np.sum(predictions[yval==0]==1)
        false_negative = np.sum(predictions[yval==1]==0)
        
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        
        F1 = (2*precision*recall)/(precision + recall)
        
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon,  bestF1
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)
print("Best epsilon found using cross-validation:",epsilon)
print("Best F1 on Cross Validation Set:",F1)
#%%
plt.figure(figsize=(8,6))

plt.scatter(X[:,0],X[:,1],marker="x")

X1,X2 = np.meshgrid(np.linspace(0,35,num=70),np.linspace(0,35,num=70))
p2 = multivariateGaussian(np.hstack((X1.flatten()[:,np.newaxis],X2.flatten()[:,np.newaxis])), mu, sigma2)
contour_level = 10**np.array([np.arange(-20,0,3,dtype=np.float)])[0]
plt.contour(X1,X2,p2[:,np.newaxis].reshape(X1.shape),contour_level)

outliers = np.nonzero(p<epsilon)[0]
plt.scatter(X[outliers,0],X[outliers,1],marker ="o",facecolor="none",edgecolor="r",s=70)

plt.xlim(0,35)
plt.ylim(0,35)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
#%%        REALISTIC DATASET NOW


mat2 = loadmat("ex8data2.mat")
X2 = mat2["X"]
Xval2 = mat2["Xval"]
yval2 = mat2["yval"]
mu2, sigma2_2 = estimateGaussian(X2)
#%%
p3 = multivariateGaussian(X2, mu2, sigma2_2)

pval2 = multivariateGaussian(Xval2, mu2, sigma2_2)

epsilon2, F1_2 = selectThreshold(yval2, pval2)
print("Best epsilon found using cross-validation:",epsilon2)
print("Best F1 on Cross Validation Set:",F1_2)
print("# Outliers found:",np.sum(p3<epsilon2))
