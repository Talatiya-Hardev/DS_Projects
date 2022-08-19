# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:00:20 2022

@author: Hardev
"""

import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
mat = scipy.io.loadmat('ex6data1.mat')
mat.keys()
#dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
plt.figure(figsize=(7,5))
ax = sns.scatterplot(x=mat['X'][:,0], y=mat['X'][:,1], hue=mat['y'].ravel(), style=mat['y'].ravel(), s=80, legend=False)
plt.title('Example Dataset 1')
plt.show(ax)
#%%

def plot_boundary(X, y, model, title):
    ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, s=80, legend=False)
    ax.set(title=title)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(xy).reshape(XX.shape)
    a = ax.contour(XX, YY, Z, colors='g', levels=[0], linestyles=['--'])
    
#%%    
from sklearn import svm

c_vals = [1.0, 100.0, 1000.0]
plt.figure(figsize=(20,5))

for i, c in enumerate(c_vals):
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(mat['X'], mat['y'].ravel())
    
    plt.subplot(1,3,i+1)
    plot_boundary(mat['X'], mat['y'].ravel(), clf, 'C={}'.format(c))
    
plt.show()
#%%
def gaussian_kernel(x1, x2, sigma=0.1):
    x1 = x1.ravel()
    x2 = x2.ravel()
    sumx1x2 = np.sum((x1-x2)**2)
    return np.exp(-sumx1x2/(2*sigma**2))
x1 = np.array([1,2,1])
x2 = np.array([0,4,-1])
sigma = 2
gk = gaussian_kernel(x1, x2, sigma)
print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {0} : {1}\n\t(for sigma = 2, this value should be about 0.324652)\n'.format(sigma, gk))
#%%      DATA SET 2

mat = scipy.io.loadmat('ex6data2.mat')
plt.figure(figsize=(7,5))
ax = sns.scatterplot(x=mat['X'][:,0], y=mat['X'][:,1], hue=mat['y'].ravel(), style=mat['y'].ravel(), s=80, legend=False)
plt.title('Example Dataset 2')
#%%
def gaussian_kernel_matrix(X1, X2, sigma=0.1):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = gaussian_kernel(x1, x2, sigma)
    return gram_matrix
#%%
def plot_gaussian_boundary(X, y, model, title='SVM Decision Boundary for Gaussian Kernel'):
    plt.figure(figsize=(7,5))
    ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, s=80, legend=False)
    ax.set(title=title)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    gram = gaussian_kernel_matrix(xy, X)
    Z = model.predict(gram).reshape(XX.shape)

    a = ax.contour(XX, YY, Z, colors='g', levels=[0.5], linestyles=['--'])
    plt.show()
    
clf = svm.SVC(kernel="precomputed", C=1.0, verbose=True)
gram = gaussian_kernel_matrix(mat['X'], mat['X'], sigma=0.1)
clf.fit(gram, mat['y'].ravel())
#%%
plot_gaussian_boundary(mat['X'], mat['y'].ravel(), clf)
#%% DATASET 3
mat = scipy.io.loadmat('ex6data3.mat')
plt.figure(figsize=(7,5))
ax = sns.scatterplot(x=mat['X'][:,0], y=mat['X'][:,1], hue=mat['y'].ravel(), style=mat['y'].ravel(), s=80, legend=False)
plt.title('Example Dataset 3')
plt.show(ax)
#%%

def find_best_c_sigma(X, y, Xval, yval):

    C_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    error = 9999

    C = 0.01
    sigma = 0.01
    for c in C_vals:
        for s in sigma_vals:
            clf = svm.SVC(kernel="precomputed", C=c)
            gram = gaussian_kernel_matrix(X, X, sigma=s)
            clf.fit(gram, y)

            gram_pred = gaussian_kernel_matrix(Xval, X)
            y_pred = clf.predict(gram_pred)
            
            error_mean = np.mean(y_pred != yval)
            if error_mean < error:
                C = c
                sigma = s
                error = error_mean
    return C, sigma

#%%
C, sigma = find_best_c_sigma(mat['X'], mat['y'].ravel(), mat['Xval'], mat['yval'].ravel())
clf = svm.SVC(kernel="precomputed", C=C)
gram = gaussian_kernel_matrix(mat['X'], mat['X'], sigma=sigma)
clf.fit(gram, mat['y'].ravel())
#%%
plot_gaussian_boundary(mat['X'], mat['y'].ravel(), clf)



#%%


#######################################################################
#SPAM CLASSIFICATION

f = open('emailSample1.txt', 'r')
contents = f.read()
#%%
import re
import string
from nltk.stem import PorterStemmer

def process_email(email_contents):
    processed = email_contents.lower()
    processed = re.sub('<[^<>]+>', ' ', processed)
    processed = re.sub('[0-9]+', 'number', processed)
    processed = re.sub('(http|https)://[^\s]*', 'httpaddr', processed)
    processed = re.sub('[^\s]+@[^\s]+', 'emailaddr', processed)
    processed = re.sub('[$]+', 'dollar', processed)
    
    for punctuation in string.punctuation:
        processed = processed.replace(punctuation, ' ')
        
    stemmer = PorterStemmer()
    processed = ' '.join([stemmer.stem(re.sub('[^a-zA-Z0-9]', '', word)) for word in processed.split()])
    processed = ' '.join(processed.split())    
    return processed.strip()
print(contents)
#%%
process_email(contents)
#%%
import re
import string
from nltk.stem import PorterStemmer

def process_email(email_contents):
    processed = email_contents.lower()
    processed = re.sub('<[^<>]+>', ' ', processed)
    processed = re.sub('[0-9]+', 'number', processed)
    processed = re.sub('(http|https)://[^\s]*', 'httpaddr', processed)
    processed = re.sub('[^\s]+@[^\s]+', 'emailaddr', processed)
    processed = re.sub('[$]+', 'dollar', processed)
    
    for punctuation in string.punctuation:
        processed = processed.replace(punctuation, ' ')
        
    stemmer = PorterStemmer()
    processed = ' '.join([stemmer.stem(re.sub('[^a-zA-Z0-9]', '', word)) for word in processed.split()])
    processed = ' '.join(processed.split())    
    return processed.strip()
#%%
df_vocab = pd.read_csv('vocab.txt', sep='\t', header=None)
df_vocab.columns = ['index', 'word']
df_vocab.sample(5)
#%%
def process_email_and_get_indices(text):
    text = process_email(text)
    print('\n======= Processed Email =======\n', text, '\n========\n')
    df_vocab = pd.read_csv('vocab.txt', sep='\t', header=None)
    df_vocab.columns = ['index', 'word']
    indices = [df_vocab[df_vocab.word==word]['index'].values[0] for word in text.split() if len(df_vocab[df_vocab.word==word]['index'].values > 0)]
    return indices
word_indices = process_email_and_get_indices(contents)
print(word_indices)
#%% EXTRACTING FEATURES FROM EMAIL
def email_features(indices):
    n = 1899
    x = np.zeros((n,1))
    x[indices]=1
    return x
features = email_features(word_indices)
print('Length of feature vector:', len(features))
print('Number of non-zero entries:', sum(features==1)[0])

#%%
data = scipy.io.loadmat('spamTrain.mat')
data_test = scipy.io.loadmat('spamTest.mat')
clf = svm.SVC(kernel='linear', C=0.1)
clf.fit(data['X'], data['y'].ravel())

print('Training Accuracy:', clf.score(data['X'], data['y'].ravel()))
print('Test Accuracy:', clf.score(data_test['Xtest'], data_test['ytest'].ravel()))
#%%