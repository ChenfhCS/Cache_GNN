from re import X
import numpy as np
import torch
import matplotlib.pyplot as plt

from kmeans_pytorch import kmeans
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

# sklearn version
def clustering(input_features, method):
    if method == 'K-mean':
        from sklearn.cluster import KMeans
        X = input_features.numpy()
        kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
        return kmeans
        # plot_clusters(X, y)

def reset_features(input_features, cluster_result):
    new_features = torch.FloatTensor(input_features.size(0), input_features.size(1))
    cluster_labels = torch.LongTensor(cluster_result.labels_)
    cluster_features = torch.tensor(cluster_result.cluster_centers_)
    new_features = cluster_features[cluster_labels]
    return new_features

# # pytorch version
# def clustering(input_features, num_clusters, method):
#     if method == 'K-mean':
#         cluster_ids_x, cluster_centers = kmeans(X=input_features, num_cluster=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
#         return cluster_ids_x, cluster_centers
#         # plot_clusters(X, y)

# def reset_features(input_features, cluster_ids_x, cluster_centers):
#     new_features = torch.FloatTensor(input_features.size(0), input_features.size(1))
#     new_features = cluster_centers[cluster_ids_x]
#     return new_features


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
