import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
from scipy.sparse import csr_matrix, find
import pandas as pd

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

data_home = './project3dataset/hw3dataset'
hits = SparseHITS()
hits.load_graph_dataset(data_home, is_undirected=False)
print("The number n of nodes: {}".format(hits.n))
print("The number m of edges: {}".format(hits.m))