import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
from scipy.sparse import csr_matrix, find
import pandas as pd
import time
tStart=time.time()

class SparseHITS:
    def load_graph_dataset(self, data_home, is_undirected=False):
        '''
        Load the graph dataset from the given directory (data_home)

        inputs:
            data_home: string
                directory path conatining a dataset (edges.tsv, node_labels.tsv)
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
        
        # Step 3. convert the edge list to the weighted adjacency matrix
        rows = edges[:, 0]
        cols = edges[:, 1]
        weights = np.ones((int(edges.shape[0]),), dtype=float)
        self.A = csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=float)
        if is_undirected == True:
            self.A = self.A + self.A.T
        self.AT = self.A.T
                
        # Step 4. set n (# of nodes) and m (# of edges)
        self.n = self.A.shape[0]     # number of nodes
        self.m = self.A.nnz          # number of edges


    def iterate_HITS(self, epsilon=1e-9, maxIters=100):
        '''
        Iterate the HITS equation to obatin the hub & authority score vectors
        
        inputs:
            epsilon: float
                the error tolerance of the iteration
            maxIters: int
                the maximum number of iterations
                
        outputs:
            h: np.ndarray (n x 1 vector)
                the final hub score vector
            a: np.ndarray (n x 1 vector)
                the final authority score vector
            h_residuals: list
                the list of hub residuals over the iteration
            a_residuals: list
                the list of authority residuals over the iteration

        '''
        old_h = np.ones(self.n)/self.n
        old_a = np.ones(self.n)/self.n
        h_residuals = []
        a_residuals = []
        
        for t in range(maxIters):
            h = self.A.dot(old_a)
            a = self.AT.dot(h)
                        
            h = h / np.linalg.norm(h, 2)
            a = a / np.linalg.norm(a, 2)
            
            h_residual = np.linalg.norm(h - old_h, 1)
            a_residual = np.linalg.norm(a - old_a, 1)
            h_residuals.append(h_residual)
            a_residuals.append(a_residual)
            old_h = h
            old_a = a
            
            if h_residual < epsilon and a_residual < epsilon:
                break
        
        return h, a, h_residuals, a_residuals

    def rank_nodes(self,ranking_scores, topk=-1):
        sorted_nodes = np.flipud(np.argsort(ranking_scores))
        sorted_scores = ranking_scores[sorted_nodes]
        ranking_results = pd.DataFrame()
        ranking_results["node_id"] = sorted_nodes
        ranking_results["score"] = sorted_scores
        
        return ranking_results[0:topk]



data_home = './project3dataset/hw3dataset'
hits = SparseHITS()
hits.load_graph_dataset(data_home, is_undirected=False)
print("The number n of nodes: {}".format(hits.n))
print("The number m of edges: {}".format(hits.m))

h, a, _, _ = hits.iterate_HITS(epsilon=1e-9, maxIters=100)

print("Top-{} rankings based on the hub score vector:".format(hits.n))
print(hits.rank_nodes(h, hits.n))

print("Top-{} rankings based on the authority score vector".format(hits.m))
print(hits.rank_nodes(a, hits.m))


tEnd=time.time()
print('Overall processing time: '+ str ( round( (tEnd-tStart) , 3) )+' seconds' )