import numpy as np
from sklearn.cluster import KMeans
import math

def onmtf(x, d1, d2, conv, verbose=False):
    """ Orthogonal Nonnegative Matrix Tri-Factorization
    minimize || X - F S G^T ||
    """
    s = np.zeros((d1, d2))

    # initialize f
    kmeans_row = KMeans(n_clusters=d1, n_jobs=-1)    
    kmeans_row.fit(x)
    mem_row = kmeans_row.predict(x)
    f = np.zeros((x.shape[0], d1))
    f[np.arange(f.shape[0]),mem_row] = 1
    f = f + 0.2
    
    # initialize g
    kmeans_col = KMeans(n_clusters=d2, n_jobs=-1)
    kmeans_col.fit(x.T)
    mem_col = kmeans_col.predict(x.T)
    g = np.zeros((d2, x.shape[1]))
    g[mem_row, np.arange(f.shape[1])] = 1
    g = g + 0.2

    # initialize s = F^T X G
    s = np.dot(np.dot(f.T, x), g)

    d = math.inf
    while d > conv ** 2:
        gprev = np.array(g)
        fprev = np.array(f)
        sprev = np.array(s)
        
        g = g * np.dot(np.dot(x.T, f), s) / np.dot(g, np.dot(g.T, np.dot(x.T, np.dot(f, s))))
        f = f * np.dot(np.dot(x, g), s.T) / np.dot(f, np.dot(f.T, np.dot(x, np.dot(g, s.T))))
        s = s * np.dot(np.dot(f.T, x), g) / np.dot(f.T, np.dot(f, np.dot(s, np.dot(g.T, g))))

        d = np.linalg.norm(gprev - g) +
            np.linalg.norm(fprev - f) +
            np.linalg.norm(sprev - s) +

        if verbose:
            print('d =', d)

    return f, s, g


def construct_codebook(x, n_clusteru, n_clusteri, codebook_conv):
    u, s, v = onmtf(x, n_clusteru, n_clusteri, codebook_conv)
    u[np.where(u > 0)] = 1
    v[np.where(v > 0)] = 1
    return np.dot(u.T, np.dot(x, v)) / np.dot(u.T, np.dot(np.ones(s.shape), v))


def transfer_codebook(x, b, n_iter):
    """
    args:
        x: target rating matrix
        b: codebook
    """
    u = np.zeros(x.shape[0], b.shape[0])
    v = np.zeros(x.shape[1], b.shape[1])
    v[np.arange(v.shape[0]), np.random.random_integers(0, v.shape[1], v.shape[0])] = 1

    mask = np.zeros(x.shape)
    mask[np.where(x > 0)] = 1

    prev_v = np.array(v)
    for t in range(n_iter):
        u = np.zeros(u.shape)
        for i in range(u.shape[0]):
            j = np.argmin(np.linalg.norm((x[i,:] - np.dot(b, prev_v.T)) * mask[i,:], axis=1))
            u[i,j] = 1
            
        v = np.zeros(v.shape)
        for i in range(v.shape[0]):
            j = np.argmin(np.linalg.norm((x[:,i] - np.dot(u, b)) * mask[:,i], axis=1))
            v[i,j] = 1
            
    return x + (1 - mast) * np.norm(u, np.norm(b, v.T))
