import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import pandas as pd
import matplotlib
import time
tStart=time.time()

class DensePageRank:
    def load_graph_dataset(self, data_home, is_undirected=False):
        '''
        Load the graph dataset from the given directory (data_home)

        inputs:
            data_home: string
                directory path conatining a dataset
            is_undirected: bool
                if the graph is undirected
        '''
        # Step 1. set file paths from data_home
        edge_path = "{}/graph_1.txt".format(data_home)

        # Step 2. read the list of edges from edge_path
        edges = np.loadtxt(edge_path, dtype=str)
        #print('type(edges)=', type(edges),'\n')
        #print('edges.shape=', edges.shape,'\n')
        new_edges=[]
        for row in edges:
            #print('\n 1 row=', row)
            row=row.split(',')
            row[0]=int(row[0])
            row[1]=int(row[1])
            #print('\n 2 row=', row)
            new_edges.append(row)
        #print('edges=', edges)
        #print('new_edges=', new_edges)
        edges=np.array( new_edges)
        #print('type(edges)=', type(edges),'\n')
        #print('edges.shape=', edges.shape,'\n')
        n = int(np.amax(edges[:, :]))+1 # the current n is the maximum node id (starting from 0)

        # Step 3. convert the edge list to the adjacency matrix
        self.A = np.zeros((n, n))
        for i in range(edges.shape[0]):
            weight=1
            #source, target, weight = edges[i, :]
            source, target=edges[i,:]
            self.A[(source, target)] = weight
            if is_undirected:
                self.A[(target, source)] = weight

        # Step 4. set n (# of nodes) and m (# of edges)
        self.n = n                         # number of nodes
        self.m = np.count_nonzero(self.A)  # number of edges

    def normalize(self):
        '''
        Perform the row-normalization of the given adjacency matrix
        '''
        # Step 1. obatin the out-degree vector d
        d = self.A.sum(axis = 1)           # row-wise summation

        # Step 2. obtain the inverse of the out-degree matrix
        d = np.maximum(d, np.ones(self.n)) # handles zero out-degree nodes, `maximum` perform entry-wise maximum 
        invd = 1.0 / d                # entry-wise division
        invD = np.diag(invd)          # convert invd vector to a diagonal matrix

        # Step 3. compute the row-normalized adjacency matrix
        self.nA = invD.dot(self.A)   # nA = invD * A
        self.nAT = self.nA.T         # nAT is the transpose of nA
        
        self.out_degrees = d

    def iterate_PageRank(self, b=0.15, epsilon=1e-9, maxIters=100):
        '''
        Iterate the PageRank equation to obatin the PageRank score vector

        inputs:
            b: float (between 0 and 1)
                the teleport probability
            epsilon : float
                the error tolerance of the iteration
            maxIters : int
                the maximum number of iterations

        outputs:
            p: np.ndarray (n x 1 vector)
                the final PageRank score vector
            residuals: list
                the list of residuals over the iteration
        '''
        q = np.ones(self.n)/self.n     # set the query vector q
        old_p = q                      # set the previous PageRank score vector
        residuals = []                 # set the list for residuals over iterations

        for t in range(maxIters):
            p = (1 - b) * (self.nAT.dot(old_p)) + (b * q)
            residual = np.linalg.norm(p - old_p, 1)
            residuals.append(residual)
            old_p = p

            if residual < epsilon:
                break

        return p, residuals


data_home = './project3dataset/hw3dataset'
dpr = DensePageRank()
dpr.load_graph_dataset(data_home, is_undirected=False)
dpr.normalize()
p, residuals = dpr.iterate_PageRank(b=0.15, epsilon=1e-9, maxIters=100)

# print the number of nodes and edges
print("The number n of nodes: {}".format(dpr.n))
print("The number m of edges: {}".format(dpr.m))

# check the sum of each row in the row-normalized matrix nA
row_sums = dpr.nA.sum(axis=1)
for (i, degree, row_sum) in zip(range(dpr.n), dpr.out_degrees, row_sums):
    print("node: {:2d}, out-degree: {:2d},  row_sum: {:.2f}".format(i, int(degree), row_sum))

# print the PageRank score of each node
for (i, score) in zip(range(dpr.n), p):
    print("node: {:2d}, PageRank score: {:.4f}".format(i, score))


tEnd=time.time()
print('Overall processing time: '+ str ( round( (tEnd-tStart) , 3) )+' seconds' )