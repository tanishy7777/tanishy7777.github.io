---
layout: post
title: Timeseries Clustering using Diffusion Maps
date: 2025-03-26 02:10:00 +5:30
description: A brief post testing various dimensionality reduction techniques for evaluating Timeseries clustering using K-means.
tags: algorithms
categories: algorithms
thumbnail: assets/img//diffusion%20maps/markov_chains.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---
In this blog post we will explore diffusion maps for Timeseries Clustering
on the UCI-HAR dataset[1].

## Understanding Diffusion maps
Diffusion maps are a nonlinear dimensionality reduction technique that models a dataset as a Markov chain, where the probability of moving from one point to another is based on their similarity(we get this
from the diffusion kernel which we explore in this blog). This is done by simulating random walks on the dataset, (called the diffusion process).


**Glossary**:

Markov Chain: If you are familiar with Finite State automata this is very similar to that.
It is a system that moves between different states step by step, where the next step depends only on the current state (not past history).

![Image of markov chains](/assets/img//diffusion%20maps/markov_chains.png)
[2]

Random Walk: It's a type of Markov Chain where we randomly move from one place to another based on probabilities.

For diffusion maps, the probabilities are based on the distances, closer points
have higher transition probability.

Consider n datapoints, we can create a transition matrix P where $P_{ij}$ denotes the probability of sum of all paths going from point $x_i$ to $x_j$
in the dataset X.

This transition probability can be given by the Diffusion kernel, where d(xi, xj) is the DTW distance in case of time series data.

$K_{ij} = \exp \left(-\frac{d(x_i, x_j)^2}{\varepsilon} \right)$


Using the **Perron-Frobenius Theorem**:
Which states that
Any matrix M that is positive (aij > 0), column stochastic matrix ($\sum_{i} a_{ij}$ = 1 ∀j)

Hence, We can say that the matrix P gives 1 as the largest eigen value and all other eigen values are <1.


Another thing is if we take $P^t$ then the elements $P_{ij}$ represent the probability of all paths of length t going from $x_i$ to $x_j$

Algorithm taken from [3] which was presented in [4]
![image.png](/assets/img/diffusion%20maps/algorithm.png)

### Defining Diffusion Distance between $x_i$ and $x_j$

$$ D_{t}(x_{i}, x_{j})^2 = \sum_{k}(|P_{ik}^t - P_{jk}^t|)^2 $$

Note, we do not need to calculate P to find the diffusion distance
It can be shown that,

$$ |Y_{i}-Y_{j}|^2 = D_{t}(x_{i}, x_{j})^2 $$

where $Y_{i} = (P_{i1}^t, P_{i2}^t, ..., P_{in}^t).T $

$Y_{i}$ is a vector T represents transpose

If we use diffussion distance in the original dataspace as a measure of similarity
then when we project this data in the diffusion/embedding space the similarity is preserved as
noted in the above equation.

Note that we can take the first k elements(excluding the 1st eigen vector) from the vector $Y_{i}$
In the below code I have taken k=3


## Load the dataset


```python
import pandas as pd
import numpy as np

def load_dataset():
    base_path = 'UCI-HAR Dataset/train/Inertial Signals/'
    signals = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]

    data = [pd.read_csv(f"{base_path}{signal}_train.txt", sep=r"\s+", header=None) for signal in signals]

    x = np.stack(data, axis=1)
    print(x.shape)
    return x

X_train = load_dataset()
```

    (7352, 9, 128)
    


```python
y_train = pd.read_csv('UCI-HAR Dataset/train/y_train.txt', header=None)
y_train = y_train.values.ravel()
```


```python
y_train.shape
```




    (7352,)



## Pairwise distance matrix(similarity matrix)

This specific implementation for Dimensionality reduction using Diffusion maps is adapted from this.[5]


```python
# Eucledean Pairwise distance matrix
pairwise_eucleadean_dissimilarity_matrix = np.zeros((1000, 1000))
for i in range(X_train[:1000].shape[0]):
    for j in range(X_train[:1000].shape[0]):
        pairwise_eucleadean_dissimilarity_matrix[i][j] = np.linalg.norm(X_train[i] - X_train[j])
```


```python
from aeon.distances.elastic import dtw_pairwise_distance

pairwise_dtw_dissimilarity_matrix = dtw_pairwise_distance(X_train[:1000], window=25, itakura_max_slope=0.5)
```

### Implementation Note
#### Note that in step 2 I have used the value of alpha = 1, as this gives the best scores.


```python
# kernel matrix
from scipy.linalg import eigh

similarity_matrix = np.exp(-pairwise_dtw_dissimilarity_matrix) # already squared(see aeon implementation)

D = np.diag(similarity_matrix.sum(axis = 1))   # symmetric dose not matter row or column

# P = similarity_matrix/D[:, None]    # markov transition matrix has probablity of going from i to j
# Can parallelize this step using matrix multiplication
P = np.linalg.inv(D) @ similarity_matrix

symm_P = (P + P.T)/2

eigenvalues, eigenvectors = np.linalg.eig(symm_P) # since symmetric matrix we use eigh


# Implementation Note, since P is not symmetric we get complex eigenvectors
# also the scores from clustering are not good so I converted P to a symmetric matrix.
# But when I kept P as it is the scores were not that good
# eigenvalues, eigenvectors = np.linalg.eig(P) # since symmetric matrix we use eigh
# ARI: 0.03900228844543332 Silhouette Score: 0.6649768743495169 if I kept P as it is

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

def compute_diffusion_coords(eigenvectors, eigenvalues, n_components, time=1):
    # Considering random walk with 1 step(focuses on local structure    )
    # Increasing t means diffusion will happen with t steps
    # this smoothens the data and focuses on global structure
    # return eigenvectors @ np.diag(np.exp(-time * eigenvalues))[:, 1:n_components+1]
    # print(eigenvectors.shape, eigenvalues.shape)
    # print(eigenvectors[:, 1:n_components+1].shape, eigenvalues[1:n_components+1].shape)
    # print((eigenvectors[:, 1:n_components+1] * eigenvalues[1:n_components+1]).shape)
    return eigenvectors[:, 1:n_components+1] * (eigenvalues[1:n_components+1]**time)
```


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.linalg import eigh
def clustering_and_evaluation(embedding, true_labels, method='kmeans'):

    if method == 'kmeans':
        k = len(np.unique(true_labels))
        clustering_model = KMeans(n_clusters=k, random_state=42)
    elif method == 'dbscan':
        clustering_model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError("Unsupported clustering method")
    print(embedding.shape)
    cluster_labels = clustering_model.fit_predict(embedding)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    sil_score = silhouette_score(embedding, cluster_labels)
    return cluster_labels, ari, sil_score
```


```python
def flatten_data(X):
    return X.reshape(X.shape[0], -1)

def compute_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)

def compute_tsne(X, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(X)

def plot_embedding(embedding, labels, title):
    plt.figure(figsize=(8,6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()
```

## Clustering in the Diffusion Space

### time = 1 seems to be the best choice


```python
# eigenvalues, eigenvectors = compute_eign_values(pairwise_dtw_dissimilarity_matrix, alpha=1)
diffusion_coords = np.real(compute_diffusion_coords(eigenvectors, eigenvalues, n_components=3 , time=1))
print("Clustering in diffusion space using KMeans...")
cluster_labels_diff, ari_diff, sil_diff = clustering_and_evaluation(diffusion_coords, y_train[:1000], method='kmeans')
print("Diffusion Map - ARI:", ari_diff, "Silhouette Score:", sil_diff)
```

    Clustering in diffusion space using KMeans...
    (1000, 3)
    Diffusion Map - ARI: 0.24257057367694157 Silhouette Score: 0.8309674918905087
    


```python
diffusion_coords_temp = np.real(compute_diffusion_coords(eigenvectors, eigenvalues, n_components=3 , time=50))
print("Clustering in diffusion space using KMeans...")
cluster_labels_diff, ari_diff, sil_diff = clustering_and_evaluation(diffusion_coords_temp, y_train[:1000], method='kmeans')
print("Diffusion Map - ARI:", ari_diff, "Silhouette Score:", sil_diff)
```

    Clustering in diffusion space using KMeans...
    (1000, 3)
    Diffusion Map - ARI: 0.23979010113271332 Silhouette Score: 0.8231012166699961
    


```python
diffusion_coords_temp = np.real(compute_diffusion_coords(eigenvectors, eigenvalues, n_components=3 , time=100))
print("Clustering in diffusion space using KMeans...")
cluster_labels_diff, ari_diff, sil_diff = clustering_and_evaluation(diffusion_coords_temp, y_train[:1000], method='kmeans')
print("Diffusion Map - ARI:", ari_diff, "Silhouette Score:", sil_diff)
```

    Clustering in diffusion space using KMeans...
    (1000, 3)
    Diffusion Map - ARI: 0.1977380764320845 Silhouette Score: 0.797945992663461
    

## Visualization and Interpretation


```python
# Plotting the diffusion map embeddings
plot_embedding(diffusion_coords[:, :2], y_train[:1000], "Diffusion Map Embedding (first 2 components)")

```


    
![plot 1 diffusion maps](/assets/img/diffusion%20maps/plot1.png)
    



```python
plot_embedding(diffusion_coords[:, [0, 2]], y_train[:1000], "Diffusion Map Embedding (first 2 components)")
```

![png](/assets/img/diffusion%20maps/plot2.png)


```python
X_flat = flatten_data(X_train[:1000])
print("Clustering in raw feature space...")
cluster_labels_raw, ari_raw, sil_raw = clustering_and_evaluation(X_flat, y_train[:1000], method='kmeans')
print("Raw Feature Space - ARI:", ari_raw, "Silhouette Score:", sil_raw)
```

    Clustering in raw feature space...
    (1000, 1152)
    Raw Feature Space - ARI: 0.2968409437567964 Silhouette Score: 0.13039108175504283
    


```python
# PCA embeddings
pca_coords = compute_pca(X_flat, n_components=3)
print("Clustering in PCA space...")
cluster_labels_pca, ari_pca, sil_pca = clustering_and_evaluation(pca_coords, y_train[:1000], method='kmeans')
print("PCA - ARI:", ari_pca, "Silhouette Score:", sil_pca)
```

    Clustering in PCA space...
    (1000, 3)
    PCA - ARI: 0.39081158060003784 Silhouette Score: 0.45552093767594737
    


```python
# t-SNE embeddings
tsne_coords = compute_tsne(X_flat, n_components=3)
print("Clustering in t-SNE space...")
cluster_labels_tsne, ari_tsne, sil_tsne = clustering_and_evaluation(tsne_coords, y_train[:1000], method='kmeans')
print("t-SNE - ARI:", ari_tsne, "Silhouette Score:", sil_tsne)
```

    Clustering in t-SNE space...
    (1000, 3)
    t-SNE - ARI: 0.25734057971635116 Silhouette Score: 0.30087483
    


```python
# Plot the alternative embeddings(PCA, tSNE)
plot_embedding(pca_coords[:, :2], y_train[:1000], "PCA Embedding")
plot_embedding(tsne_coords, y_train[:1000], "t-SNE Embedding")
```

![png](/assets/img/diffusion%20maps/plot3.png)

    

![png](/assets/img/diffusion%20maps/plot4.png)

    


## Observations

### Adjusted Rand Index (ARI):
ARI measures how well the clustering results match the known ground truth labels. Here, PCA (ARI ≈ 0.391) outperforms the other methods, meaning that clusters obtained in the PCA space align best with the activity labels. Raw feature space (ARI ≈ 0.297) and t-SNE (ARI ≈ 0.257) follow, with diffusion maps having the lowest ARI (≈ 0.108). This suggests that, in terms of matching the provided labels, PCA produces clusters that most closely reflect the expected classes.

### Silhouette Score:
The silhouette score reflects the quality of the clustering, how **well-separated and compact the clusters** are.

Diffusion maps dominate in this respect with a high silhouette score of **0.839**, indicating that within the diffusion space, the clusters are very well-defined and separated, even though they don’t align as well with the ground truth labels. 

PCA also shows good enough separation  **0.456**, while raw features shows **0.130** and t-SNE **0.301** are lower.

TODO: # Add legend/labels in the plots

## Explorative
TODO: 
1. Implement Multiscale Diffusion Maps to capture hierarchical structures in time-series data.
2. Use Spectral Clustering instead of K-Means for improved grouping.


## References Used:


[1]“UCI Machine Learning Repository,” archive.ics.uci.edu. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

[2] Wikipedia Contributors, “Markov chain,” Wikipedia, Dec. 13, 2019. https://en.wikipedia.org/wiki/Markov_chain


[3] Wikipedia Contributors, “Diffusion map,” Wikipedia, Jan. 03, 2025.


[4] Coifman, R.R.; Lafon, S; Lee, A B; Maggioni, M; Nadler, B; Warner, F; Zucker, S W (2005). 

[5] NPTEL IIT Guwahati, “Lec 49: Diffusion maps,” YouTube, Mar. 31, 2022. https://www.youtube.com/watch?v=SaqJ8x4vQ6U (accessed Mar. 25, 2025).

‌
‌
