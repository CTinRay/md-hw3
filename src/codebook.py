import numpy as np
from sklearn.cluster import KMeans
import math
import argparse
import random
from sklearn.decomposition import PCA

def load_matrix(shape, filename):
    matrix = np.zeros(shape)
    f = open(filename)
    for l in f:
        cs = l.strip().split(' ')
        x = int(cs[0])
        y = int(cs[1])
        v = float(cs[2])
        matrix[x][y] = v

    return matrix


def onmtf(x, d1, d2, conv, verbose=False):
    """ Orthogonal Nonnegative Matrix Tri-Factorization
    minimize || X - F S G^T ||
    """
    s = np.zeros((d1, d2))

    # print(np.sum(x > 0, axis=1))
    # many1 = np.where(np.sum(x > 0, axis=1) > (x.shape[1] / 2))[0]
    # print('# of rows with many 1 =', many1.shape[0])
    
    # initialize f
    # kmeans_row = KMeans(n_clusters=d1, n_jobs=-1, max_iter=10)    
    # kmeans_row.fit(x)
    # mem_row = kmeans_row.predict(x)
    f = np.random.random((x.shape[0], d1))
    # mem_row = np.random.random_integers(0, f.shape[1] - 1, f.shape[0])
    # f[np.arange(f.shape[0]),mem_row] = 1
    # f = f + 0.2
    if verbose:
        print('finish initialize f')
    
    # initialize g
    # kmeans_col = KMeans(n_clusters=d2, n_jobs=-1, max_iter=10)
    # kmeans_col.fit(x.T)
    # mem_col = kmeans_col.predict(x.T)
    g = np.random.random((x.shape[1], d2))
    # mem_col = np.random.random_integers(0, g.shape[1] - 1, g.shape[0])
    # g[np.arange(g.shape[0]), mem_col] = 1
    # g = g + 0.2
    if verbose:
        print('finish initialize g')

    # initialize s = F^T X G
    s = np.random.random((d1, d2))

    d = math.inf

    print('g =', g[0])
    print('f =', f[0])
    print('s =', s[0])
                
    while d > conv ** 2:
        gprev = np.array(g)
        fprev = np.array(f)
        sprev = np.array(s)

        g = g * np.sqrt(np.dot(np.dot(x.T, f), s) / np.dot(g, np.dot(np.dot(g.T, x.T), np.dot(f, s))))
        print('loss =', np.linalg.norm(x - np.dot(np.dot(f, s), g.T)))
        f = f * np.sqrt(np.dot(np.dot(x, g), s.T) / np.dot(np.dot(f, np.dot(f.T,x)),np.dot(g, s.T)))
        print('loss =', np.linalg.norm(x - np.dot(np.dot(f, s), g.T)))
        s = s * np.sqrt(np.dot(np.dot(f.T, x), g) / np.dot(np.dot(np.dot(f.T, f),s),np.dot(g.T, g)))
        print('loss =', np.linalg.norm(x - np.dot(np.dot(f, s), g.T)))

        print('g =', g[0])
        print('f =', f[0])
        print('s =', s[0])
        print('dg =', np.linalg.norm(gprev - g))        
        print('df =', np.linalg.norm(fprev - f))        
        print('ds =', np.linalg.norm(sprev - s))        
        d = np.linalg.norm(gprev - g) + np.linalg.norm(fprev - f) + np.linalg.norm(sprev - s)

        if verbose:
            print('d =', d)

    return f, s, g


def construct_codebook(x, n_clusteru, n_clusteri, codebook_conv):
    average = np.sum(x) / np.sum(x > 0)
    x_filled = np.array(x)
    x_filled[np.where(x == 0)] = average
    u, s, v = onmtf(x_filled, n_clusteru, n_clusteri, codebook_conv, verbose=True)
    u[np.where(u > 0)] = 1
    v[np.where(v > 0)] = 1
    return np.dot(u.T, np.dot(x, v)) / np.dot(u.T, np.dot(np.ones(x.shape), v))


def transfer_codebook(x, b, n_iter):
    """
    args:
        x: target rating matrix
        b: codebook
    """
    u = np.zeros((x.shape[0], b.shape[0]))
    v = np.zeros((x.shape[1], b.shape[1]))
    v[np.arange(v.shape[0]), np.random.random_integers(0, v.shape[1] - 1, v.shape[0])] = 1

    mask = np.zeros(x.shape)
    mask[np.where(x > 0)] = 1

    prev_v = np.array(v)
    for t in range(n_iter):
        u = np.zeros(u.shape)
        bvt = np.dot(b, prev_v.T)
        for i in range(u.shape[0]):
            j = np.argmin(np.linalg.norm((x[i,:] - bvt) * mask[i,:], axis=1))
            u[i,j] = 1
            
        v = np.zeros(v.shape)
        ub = np.dot(u, b)
        for i in range(v.shape[0]):
            j = np.argmin(np.linalg.norm((x[:,i].reshape(-1, 1) - ub) * mask[:,i].reshape(-1, 1), axis=1))
            v[i,j] = 1

        print('iter:', t)
            
    return x + (1 - mast) * np.norm(u, np.norm(b, v.T))


def main():
    parser = argparse.ArgumentParser(description='===== BASELINE =====')
    parser.add_argument('source', type=str, help='source.txt')
    parser.add_argument('train', type=str, help='train.txt')
    parser.add_argument('d1', type=int, help='shape of train')
    parser.add_argument('d2', type=int, help='shape of train')
    parser.add_argument('--conv', type=float, help='converge value for ONMTF', default=1000)            
    parser.add_argument('--n_iter', type=bool, help='number of iteration for codebook transfer', default=200)        
    parser.add_argument('--verbose', type=bool, help='verbose, default = False', default=False)    
    parser.add_argument('--n_clusteru', type=int, help='number of user clusters', default=100)
    parser.add_argument('--n_clusteri', type=int, help='number of item clusters', default=50)
    parser.add_argument('--holdout', type=float, help='ratio of holdout data', default=0.1)
    args = parser.parse_args()

    source = load_matrix((args.d1, args.d2), args.source)
    target = load_matrix((args.d1, args.d2), args.train)
    
    rated = [[], []]    
    rated[0] = np.array(np.where(target > 0)[0])
    rated[1] = np.array(np.where(target > 0)[1])
    holdout_inds = random.sample(range(len(rated[0])), int(len(rated[0]) * args.holdout))
    holdout_xs = rated[0][holdout_inds]
    holdout_ys = rated[1][holdout_inds]
    holdout_ans = target[holdout_xs,holdout_ys]
    target[holdout_xs,holdout_ys] = 0

    
    codebook = construct_codebook(source, args.n_clusteru, args.n_clusteri, args.conv)
    res = transfer_codebook(target, codebook, args.n_iter)
    
    holdout_res = res[holdout_xs, holdout_ys]
    accuracy = np.sum((holdout_res - holdout_ans)**2) / holdout_ans.shape[0]
    print('accuracy:', accuracy)

    

if __name__ == '__main__':
    main()
