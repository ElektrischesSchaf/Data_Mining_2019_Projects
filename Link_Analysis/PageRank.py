import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import pandas as pd
import matplotlib

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
        print('type(edges)=', type(edges),'\n')
        print('edges.shape=', edges.shape,'\n')
        new_edges=[]
        for row in edges:
            print('\n 1 row=', row)
            row=row.split(',')
            row[0]=int(row[0])
            row[1]=int(row[1])
            print('\n 2 row=', row)
            new_edges.append(row)
        print('edges=', edges)
        print('new_edges=', new_edges)
        edges=np.array( new_edges)
        print('type(edges)=', type(edges),'\n')
        print('edges.shape=', edges.shape,'\n')
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

data_home = './project3dataset/hw3dataset'
dpr = DensePageRank()
dpr.load_graph_dataset(data_home, is_undirected=False)
#dpr.load_node_labels(data_home)

# print the number of nodes and edges
print("The number n of nodes: {}".format(dpr.n))
print("The number m of edges: {}".format(dpr.m))

# print the heads (5) of the node labels
#display(dpr.node_labels.head(5))