import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


def visualize(X):
    # Visualizing the normalized sepal features
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7, edgecolors='k')
    plt.xlabel("Normalized Sepal Length")
    plt.ylabel("Normalized Sepal Width")
    plt.title("Visualization of Normalized Sepal Features in Iris Dataset")
    plt.grid(True)
    plt.show()

    # Visualizing the normalized petal features
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 2], X[:, 3], alpha=0.7, edgecolors='k')
    plt.xlabel("Normalized Petal Length")
    plt.ylabel("Normalized Petal Width")
    plt.title("Visualization of Normalized Petal Features in Iris Dataset")
    plt.grid(True)
    plt.show()


def assign_clusters(X: np.ndarray, centroids: np.ndarray):
    """
    Assigns each data point to the nearest centroid.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        centroids (ndarray): Current centroids of shape (K, n_features).

    Returns:
        labels (ndarray): Cluster assignments for each data point.
    """

    INTMAX = 1024
    dists: np.ndarray = np.empty(shape=(centroids.shape[0], 1))
    dists.fill(INTMAX)
    labels: np.ndarray = np.empty(shape=X.shape[0])

    for i in range(X.shape[0]):
        # TODO: Compute the distance of each point to all centroids
        diff: np.ndarray = centroids - X[i]
        norms = np.linalg.norm(diff, axis=1)

        # TODO: Assign each point to the closest centroid
        ind_min = np.argmin(norms)
        labels[i] = ind_min

    return labels


def initialize_centroids_kmeans_pp(X, K):
    """
    Initializes K cluster centroids using a simplified max-min method.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        K (int): Number of clusters.

    Returns:
        centroids (ndarray): Initialized centroids of shape (K, n_features).
    """
    INTMAX = 1000
    # Step 1: Select the first centroid
    centroids: np.ndarray = np.empty(shape=(K, X.shape[1]))
    centroids[0] = X[0]
    dists: np.ndarray = np.array([INTMAX for i in range(X.shape[0])])

    for _ in range(1, K):
        # Step 2: Compute distance of each point to the nearest centroid
        for centroid in centroids:
            diff = X - centroid
            norms = np.linalg.norm(diff, axis=1)
            dists = np.minimum(dists, norms)

        # Step 3: Choose the point with the max distance as the new centroid)
        ind_max = np.argmax(dists)
        centroids[_] = X[ind_max]

    print(centroids)
    return centroids


def update_centroids(X, labels, K):
    """
    Updates cluster centroids based on the mean of assigned points.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        labels (ndarray): Cluster assignments for each data point.
        K (int): Number of clusters.

    Returns:
        new_centroids (ndarray): Updated centroids of shape (K, n_features).
    """
    # TODO: Compute new centroids as the mean of assigned data points
    new_centroids = np.empty(shape=(K, X.shape[1]))
    for i in range(K):
        cluster_i = X[labels == i]
        new_centroids[i] = np.mean(cluster_i, axis=0)

    return new_centroids


def preprocess_iris(df):
    """ Preprocesses only petal features (more relevant for clustering). """
    X = np.array(df)
    X_mins = X.min(axis=0)
    X_maxs = X.max(axis=0)
    X_scaled = (X - X_mins) / (X_maxs - X_mins)

    return X_scaled


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    feature_names = iris.feature_names  # Get feature names

    # Convert to DataFrame for better visualization
    df = pd.DataFrame(X, columns=feature_names)
    print(df)

    X = preprocess_iris(df)
    K = 3

    centroids = initialize_centroids_kmeans_pp(X, K)
    labels = assign_clusters(X, centroids)

    update_centroids(X, labels, K)
