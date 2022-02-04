import numpy as np




def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, K=2, max_iters=200):
        self.K = K
        self.max_iters = max_iters

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters),self.centroids

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        return sum(distances) == 0

    def distance(self,x, cen):
        Xs=x
        cent=cen
        dis=[]
        for i in range(len(cent)):
            summ=[]
            for j in Xs[i]:
                summ.append((j-cent[i]) ** 2)
            dis.append(sum(summ))
        return dis

    def inertia_(self):
        clus=self._get_cluster_labels(self.clusters)
        cen=self.centroids
        x=[]
        uni=np.unique(clus)
        for i in range(self.K):
            a=[]
            for j in range(len(clus)):
                if clus[j]==uni[i]:
                    a.append(self.X[j].tolist())
            x.append(a)
        s=sum(self.distance(x,cen))
        return s
