import argparse
import numpy as np
import math
import random

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


class Baseline:
    def __init__(self, dlatent, reg, rate, converge, verbose):        
        self.dlatent = dlatent
        self.converge = converge
        self.verbose = verbose
        self.rate = rate
        self.reg = reg
        
    def fit(self, r):
        """ decompose rating into x^Ty """
        bu = np.zeros((r.shape[0], 1))
        bi = np.zeros((1, r.shape[1]))
        avg = np.average(r) / np.sum(np.where(r != 0))
        x = avg * np.random.random((self.dlatent, r.shape[0]))
        y = avg * np.random.random((self.dlatent, r.shape[1]))

        rated = np.where(r != 0)
        mask = np.zeros(r.shape)
        mask[rated[0],rated[1]] = 1
        
        gradient = math.inf
        
        while gradient > self.converge ** 2:
            e = (r - np.dot(x.T, y) - bu - bi - avg) * mask
            dbu = - np.sum(e, axis=1).reshape(-1, 1) + self.reg * bu
            dbi = - np.sum(e, axis=0).reshape(1, -1) + self.reg * bi
            dx = self.reg * x
            for u in range(dx.shape[0]):
                for i in range(y.shape[0]):
                    dx[:,u] += (np.dot(x[:,u].T, y[:,i]) + bu[u][0] + bi[0][i] + avg - r[u,i]) * y[:,i] * mask[u,i]

            dy = self.reg * y
            for i in range(dy.shape[0]):
                for u in range(x.shape[0]):
                    dy[:,i] += (np.dot(x[:,u].T, y[:,i]) + bu[u][0] + bi[0][i] + avg - r[u,i]) * x[:,u] * mask[u,i]

            bu -= dbu * self.rate
            bi -= dbi * self.rate
            x -= dx * self.rate
            y -= dy * self.rate
            gradient = np.linalg.norm(x) + np.linalg.norm(y) + np.linalg.norm(dbu) + np.linalg.norm(dbi)
            if self.verbose:
                print('gradient: dx =', dx, 'dy =', dy, 'dbu =', dbu, 'dbi =', dbi)
                print('|gradient|^2 =', gradient)
            
        self.x = x
        self.y = y

    def predict():
        return np.dot(self.x.T, self.y)
        
                
def main():
    parser = argparse.ArgumentParser(description='===== BASELINE =====')
    parser.add_argument('train', type=str, help='train.txt')
    parser.add_argument('d1', type=int, help='shape of train')
    parser.add_argument('d2', type=int, help='shape of train')
    parser.add_argument('--converge', type=float, help='converge, default = 0.001', default=0.001)
    parser.add_argument('--verbose', type=bool, help='verbose, default = False', default=False)    
    parser.add_argument('--reg', type=float, help='lambda for the regulization term', default=0.1)
    parser.add_argument('--dlatent', type=int, help='dimension of latent', default=100)
    parser.add_argument('--holdout', type=float, help='ratio of holdout data', default=0.1)
    parser.add_argument('--rate', type=float, help='learning rate', default=0.01)
    args = parser.parse_args()

    r = load_matrix((args.d1, args.d2), args.train)
    rated = [[], []]    
    rated[0] = np.array(np.where(r > 0)[0])
    rated[1] = np.array(np.where(r > 0)[1])
    holdout_inds = random.sample(range(len(rated[0])), int(len(rated[0]) * args.holdout))
    holdout_xs = rated[0][holdout_inds]
    holdout_ys = rated[1][holdout_inds]
    holdout_ans = r[holdout_xs,holdout_ys]
    r[holdout_xs,holdout_ys] = 0
    
    baseline = Baseline(args.dlatent, args.reg, args.rate, args.converge, args.verbose)
    baseline.fit(r)
    res = baseline.predict()
    holdout_res = res[holdout_xs, holdout_ys]
    accuracy = np.sum(np.where(holdout_res == holdout_ans)) / holdout_ans.shape[0]
    print('accuracy:', accuracy)

    
    
if __name__ == '__main__':
    main()
