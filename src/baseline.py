import argparse
import numpy as np
import math

class Baseline:
    def __init__(self, dlatent, reg, rate, converge, verbose):        
        self.dlatent = dlatent
        self.converge = converge
        self.verbose = verbose
        self.rate = rate
        self.reg = reg
        
    def fit(self, r):
        """ decompose rating into x^Ty """
        bu = np.zeros((1, r.shape[0]))
        bi = np.zeros((rating.shape[1], 1))
        avg = np.average(r)    
        x = avg * np.random.random((dlatent, r.shape[0]))
        y = avg * np.random.random((dlatent, r.shape[1]))
        
        gradient = math.inf
        
        while gradient > converge:
            e = rate - np.dot(x.T, y) - bu - bi
            dbu = - 2 * np.sum(e, axis=1) + 2 * self.reg * bu
            dbi = - 2 * np.sum(e, axis=0) + 2 * self.reg * bi
            dx = np.zeros(x.shape)
            for u in range(dx.shape[0]):
                for i in range(y.shape[0]):
                    dx[:,u] += np.dot(np.dot(x[:u].T, y[:i]) + bu[u] + bi[i] - r[u,i], y[:i]).T

            dy = np.zeros(x.shape)
            for i in range(dy.shape[0]):
                for u in range(x.shape[0]):
                    dy[:i] += np.dot(np.dot(x[:u], y[:i]) + bu[u] + bi[i] -r[u,i], x[:u]).T

            bu -= dbu
            bi -= dbi
            x -= dx
            y -= dy
            gradient = np.norm(x) + np.norm(y) + np.norm(dbu) + np.norm(dbi)

        self.x = x
        self.y = y
            
                
def main():
    parser = argparse.ArgumentParser(description='===== BASELINE =====')
    parser.add_argument('train', type=str, help='train.txt')
    parser.add_argument('--converge', type=float, help='converge, default = 0.001', default=0.001)
    parser.add_argument('--verbose', type=bool, help='verbose, default = False', default=False)    
    parser.add_argument('--reg', type=float, help='lambda for the regulization term', default=0.5)
    parser.add_argument('--dlatent', type=float, help='dimension of latent', default=0.5)
    parser.add_argument('--holdout', type=float, help='ratio of holdout data', default=0.1)
    parser.add_argument('--rate', type=float, help='learning rate', default=0.01)
    args = parser.parse()

    

    

if __name__ == '__main__':
    main()
