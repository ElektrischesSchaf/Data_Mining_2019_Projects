import nose.tools as nt
import networkx
import os
import numpy as np
import sys
#sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../..')))
from networkx_addon.similarity.simrank import simrank
from networkx_addon.similarity.simrank import simrank2
#print(dir(simrank))
'''
G = networkx.Graph()
G.add_edges_from([('1','2'), ('1', '4'), ('2','3'), ('3','1'), ('4', '5'), ('5', '4')])
print(G)
print(simrank(G))
'''

data_home = './project3dataset/hw3dataset'

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

K=networkx.Graph()
K.add_edges_from(new_edges)
print(K)
print(simrank(K))

'''
K=networkx.DiGraph()
K.add_edges_from(new_edges)
#print(K)
print(simrank2(K))
'''