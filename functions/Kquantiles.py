import numpy as np

class KQuantiles:
    """
    Class for KQuantiles++ clustering on the simplex for compositional data with zeros.
    As per the paper: "k-quantiles: L1 distance clustering under a sum constraint",
                        https://doi.org/10.1016/j.patrec.2017.03.028

    An extra initialization step is added to ensure the centers include the vertices of the simplex for better visualizations.
    """
    
    def __init__(self, n_clusters=8, random_state=None, max_iter=300, verbose=False):
        """
        random_state: None or int, Random seed for reproducibility.
        max_iter: int, Maximum number of iterations for the algorithm.

        """

        self.k = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)

        self.centers = None
        self.clusters = None

    def initialize_centers(self, X):
        """ 
        Initialize centers using k-means++ adapted for k-quantiles. 
        L1 distance is used to compute the distances between points and centers.

        Dynamic programming for minimal distances (phi) and probability computations.
        """
        n, d = X.shape
        centers = np.zeros((self.k, d))
        phi = np.zeros(n)
        l = 0

        if self.k >= d:
            for j in range(d):
                if (X[:, j] > 0.999).any():
                    centers[l, j] = 1
                    l += 1
        if l == 0:
            # One random sampling if the above initialization did not happen.
            idx = self.random_state.randint(n)
            centers[0] = X[idx]

        phi = np.min(np.sum(np.abs(X[:, np.newaxis] - centers), axis=2), axis=1)

        # Choose subsequent centers and update minimum distances
        for i in range(l, self.k):
            idx = self.random_state.choice(n, p=phi / np.sum(phi))
            centers[i] = X[idx]
            phi = np.minimum(phi, np.sum(np.abs(X - centers[i]), axis=1))
            if (phi == 0).all() and i < self.k - 1:
                print("n_clusters exceeds the number of unique points")
                print("Adjusting the number of clusters to", i)
                self.k = i
                new_centers = np.zeros((i, d))
                new_centers = centers[:i, :]
                centers = new_centers
                break
        
        self.centers = centers
        return centers

    def assign_clusters(self, X):
        """ Assign points in X to clusters based on L1 distance.
            There may be a memory issue (then convert to a for loop) 
        """
        distances = np.sum(np.abs(X[:, np.newaxis] - self.centers), axis=2)
        self.clusters = np.argmin(distances, axis=1)
        return self

    def update_centers(self, X, C=1):
        """ Update centers as per the k-quantiles method. """
        n, d = X.shape
        new_centers = np.zeros((self.k, d))
        for i in range(self.k):
            cluster_points = X[self.clusters == i]
            if len(cluster_points) <= 1:
                # No update occurs for these cases
                new_centers[i, :] = self.centers[i, :]
                continue

            # Sorting for the quantile estimation
            sorted_points = np.sort(cluster_points, axis=0)
            quantile_sum = np.sum(sorted_points, axis=1)
            
            idx = min(np.searchsorted(quantile_sum, C), len(quantile_sum) - 1) # prevent numerical errors sometimes
            if idx == 0:
                # Exception: denominator of theta becomes zero; any sorted_point works
                new_centers[i, :] = sorted_points[idx, :]
            else:
                # Computing new centroid coordinates
                theta = (quantile_sum[idx] - C) / (quantile_sum[idx] - quantile_sum[idx - 1])
                new_centers[i, :] = theta * sorted_points[idx - 1, :] + (1 - theta) * sorted_points[idx, :]
        
        self.centers = new_centers
        return self
    

    def fit(self, X, C=1):
        """ Main k-quantiles++ clustering algorithm. """
        self.initialize_centers(X)
        for _ in range(self.max_iter):
            self.assign_clusters(X)
            temp = self.centers
            self.update_centers(X, C)
            if np.allclose(temp, self.centers):
                if self.verbose:
                    print("Converged after", _ + 1, "iterations.")
                break
        #     centers = new_centers
        # return centers, clusters


def initialize_centers(X, k, init=True):
    """ 
    Initialize centers using k-means++ adapted for k-quantiles. 

    L1 distance is used to compute the distances between points and centers.
    Use dynamic programming for minimal distances (phi) and probability computations.

    """
    n, d = X.shape
    centers = np.zeros((k, d))
    phi = np.zeros(n)
    l = 1

    if init and k >= d:
        for j in range(d):
            if (X[:, j] > 0.999).any():
                centers[l - 1, j] = 1
                l += 1
    if l == 1:
        # One random sampling if the above initialization did not happen.
        idx = np.random.randint(n)
        centers[0] = X[idx]

    phi = np.min(np.sum(np.abs(X[:, np.newaxis] - centers), axis=2), axis=1)

    # Choose subsequent centers and update minimum distances
    for i in range(l, k):
        idx = np.random.choice(n, p=phi / np.sum(phi))
        centers[i] = X[idx]
        phi = np.minimum(phi, np.sum(np.abs(X - centers[i]), axis=1))

    return centers

def assign_clusters(X, centers):
    """ Assign points in X to clusters based on L1 distance.
        There may be a memory issue (then convert to a for loop) 
    """
    distances = np.sum(np.abs(X[:, np.newaxis] - centers), axis=2)
    return np.argmin(distances, axis=1)

def update_centers(X, clusters, k, C=1):
    """ Update centers as per the k-quantiles method. """
    n, d = X.shape
    new_centers = np.zeros((k, d))
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) <= 1:
            # If length == 1, then that point is just the centroid
            continue

        # Sorting for the quantile estimation
        sorted_points = np.sort(cluster_points, axis=0)
        quantile_sum = np.sum(sorted_points, axis=1)
        
        idx = np.searchsorted(quantile_sum, C)
        if idx == 0:
            # Exception: denominator of theta becomes zero.
            # quantile_sum == ones, and any sorted_points works
            new_centers[i, :] = sorted_points[idx, :]
        else:
            # Computing new centroid coordinates
            theta = (quantile_sum[idx] - C) / (quantile_sum[idx] - quantile_sum[idx - 1])
            new_centers[i, :] = theta * sorted_points[idx - 1, :] + (1 - theta) * sorted_points[idx, :]
    
    return new_centers

def k_quantiles_plus_plus(X, k, C=1, max_iter=300, verbose=True):
    """ Main k-quantiles++ clustering algorithm. """
    centers = initialize_centers(X, k)
    for _ in range(max_iter):
        clusters = assign_clusters(X, centers)
        new_centers = update_centers(X, clusters, k, C)
        if np.allclose(centers, new_centers):
            if verbose:
                print("Converged after", _ + 1, "iterations.")
            break
        centers = new_centers
    return centers, clusters


if __name__ == '__main__':

    # Example usage
    n_points = 100
    dimensions = 5
    X = np.random.rand(n_points, dimensions)  # Random dataset on a simplex
    X /= X.sum(axis=1, keepdims=True)  # Normalize to satisfy sum constraint C=1
    k = 6  # Number of clusters
    C = 1  # Sum constraint

    # centers, clusters = k_quantiles_plus_plus(X, k, C)

    kquantiles = KQuantiles(n_clusters=k, random_state=0, max_iter=300, verbose=True)
    kquantiles.fit(X, C)

    print("Centers:\n", kquantiles.centers)
    print("Cluster assignments:", kquantiles.clusters)

    # Check for different seeds and the same seed
    kquantiles = KQuantiles(n_clusters=k, random_state=5, max_iter=300, verbose=True)
    kquantiles.fit(X, C)

    print("Centers:\n", kquantiles.centers)
    print("Cluster assignments:", kquantiles.clusters)

