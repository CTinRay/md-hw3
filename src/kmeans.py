import numpy as np
from sklearn.cluster import KMeans
import argparse

def load_model(filename):
    f = open(filename, 'r')
    f.readline()
    m = int(f.readline().split()[1])
    n = int(f.readline().split()[1])
    print('m', m, 'n', n)
    k = int(f.readline().split()[1])
    f.readline()

    p = np.zeros((m, k))
    q = np.zeros((n, k))
    for i in range(m):
        l = f.readline()
        p[i] = list(map(float, l.split()[2:]))

    for i in range(n):
        l = f.readline()
        q[i] = list(map(float, l.split()[2:]))
    
    return p, q


def main():
    parser = argparse.ArgumentParser(description='===== BASELINE =====')
    parser.add_argument('source_model', type=str, help='source model', default=None)
    parser.add_argument('cluster_u', type=str, help='source-cu.txt')
    parser.add_argument('cluster_i', type=str, help='source-ci.txt')
    parser.add_argument('--n_iter', type=bool, help='number of iteration for codebook transfer', default=200)        
    parser.add_argument('--verbose', type=bool, help='verbose, default = False', default=False)    
    parser.add_argument('--n_clusteru', type=int, help='number of user clusters', default=100)
    parser.add_argument('--n_clusteri', type=int, help='number of item clusters', default=100)
    args = parser.parse_args()
    p, q = load_model(args.source_model)
    kmeans_u = KMeans(n_clusters=args.n_clusteru, n_jobs=-1, max_iter=args.n_iter, verbose=args.verbose, n_init=16)
    kmeans_u.fit(p)
    mem_u = kmeans_u.predict(p)
    kmeans_i = KMeans(n_clusters=args.n_clusteri, n_jobs=-1, max_iter=args.n_iter, verbose=args.verbose, n_init=16)
    kmeans_i.fit(q)
    mem_i = kmeans_i.predict(q)

    f = open(args.cluster_u, 'w')
    for mu in mem_u:
        f.write(str(mu) + '\n')
    f.close()

    f = open(args.cluster_i, 'w')
    for mi in mem_i:
        f.write(str(mi) + '\n')
    f.close()
    

if __name__ == '__main__':
    main()
