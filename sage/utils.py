import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from numpy import where
from dgl import DGLGraph


def Approx_prefix(input_features, parameter=0.5):
    '''
    Approx_results -> the first parameter*length values of the feature
    '''
    scale = int(input_features.size(1)*parameter)
    approx_results = input_features[:, :scale]
    return approx_results

def clustering(input_features, method):
    if method == 'K-mean':
        from sklearn.cluster import KMeans
        X = input_features.numpy()
        kemans = KMeans(n_clusters=7, random_state=0).fit(X)
        y = kemans.predict(X)
        plot_clusters(X, y)

def plot_clusters(X, cluster_ids):
    '''
    TSNE aims to decrease the dimension of X
    '''
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    x = tsne.fit_transform(X)

    for class_value in range(3):
        row_ix = where(cluster_ids == class_value)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()
