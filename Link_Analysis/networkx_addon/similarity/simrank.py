"""
Implement SimRank similarity https://github.com/hhchen1105/networkx_addon/blob/master/similarity/simrank.py
"""
#        Copyright (C) 2004-2010 by
#        Hung-Hsuan Chen <hhchen@psu.edu>
#        All rights reserved.
#        BSD license.
#        NetworkX:http://networkx.lanl.gov/.
import copy
import sys
import numpy
import itertools
import networkx as nx
from collections import defaultdict
__author__ = """Hung-Hsuan Chen (hhchen@psu.edu)"""
__all__ = ['simrank']

def simrank(G, c=0.9, max_iter=100, remove_neighbors=False, remove_self=False, dump_process=False):
    """Return the SimRank similarity between nodes

    Parameters
    -----------
    G : graph
        A NetworkX graph
    c : float, 0 < c <= 1
        The number represents the relative importance between in-direct neighbors
        and direct neighbors
    max_iter : integer
        The number specifies the maximum number of iterations for simrank
        calculation
    remove_neighbors: boolean
        if true, the similarity value between neighbor nodes is set to zero
    remove_self : boolean
        if true, the similarity value between a node and itself is set to zero
    dump_process: boolean
        if true, the calculation process is dumped

    Returns
    -------
    simrank: dictionary of dictionary of double
        if simrank[i][j] = k, this means the SimRank similarity
        between node i and node j is k

    Examples
    --------
    >>> G=nx.Graph()
    >>> G.add_edges_from([(0,7), (0,1), (0,2), (0,3), (1,4), (2,4), (3,4), (4,5), (4,6)])
    >>> networkx_addon.similarity.simrank(G)

    Notes
    -----

    References
    ----------
    [1] G. Jeh and J. Widom.
    SimRank: a measure of structural-context similarity.
    In KDD'02 pages 538-543. ACM Press, 2002.
    """
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("simrank() not defined for graphs with multiedges.")

    if G.is_directed():
        raise Exception("simrank() not defined for directed graphs.")

    sim_old = defaultdict(list)
    sim = defaultdict(list)
    for n in G.nodes():
        sim[n] = defaultdict(int)
        sim[n][n] = 1
        sim_old[n] = defaultdict(int)
        sim_old[n][n] = 0

    # calculate simrank
    for iter_ctr in range(max_iter):
        if _is_converge(sim, sim_old):
            break
        sim_old = copy.deepcopy(sim)
        for i, u in enumerate(G.nodes()):
            if dump_process:
                sys.stdout.write("\r%d : % d / %d" % (iter_ctr, i, G.number_of_nodes()))
            for v in G.nodes():
                if u == v:
                    continue
                s_uv = 0.0
                for n_u in G.neighbors(u):
                    for n_v in G.neighbors(v):
                        s_uv += sim_old[n_u][n_v]
                sim[u][v] = (c * s_uv / (len( list(G.neighbors(u))) * len( list(G.neighbors(v))))) \
                        if len( list( G.neighbors(u) )) * len(list(G.neighbors(v))) > 0 else 0
        if dump_process:
            print (' ')

    if remove_self:
        for m in G.nodes():
            G[m][m] = 0

    if remove_neighbors:
        for m in G.nodes():
            for n in G.neighbors(m):
                sim[m][n] = 0
    return sim

def _is_converge(s1, s2, eps=1e-4):
    for i in s1.keys():
        for j in s1[i].keys():
            if abs(s1[i][j] - s2[i][j]) >= eps:
                return False
    return True


# https://stackoverflow.com/questions/9767773/calculating-simrank-using-networkx
def simrank2(G, r=0.8, max_iter=100, eps=1e-4):

    nodes = G.nodes()
    print(type(nodes))
    nodes=list(nodes)
    print('nodes= ', nodes)
    nodes_i = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}
    print('nodes_i=', nodes_i,'\n')
    sim_prev = numpy.zeros(len(nodes))
    sim = numpy.identity(len(nodes))

    for i in range(max_iter):
        if numpy.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = numpy.copy(sim)
        for u, v in itertools.product(nodes, nodes):
            if u is v:
                continue
            u_ns, v_ns = G.predecessors(u), G.predecessors(v)
            #print('len(u_us)= ',len(list(u_ns)))
            u_ns_len=len(list(u_ns))
            v_ns_len=len(list(v_ns))
            # evaluating the similarity of current iteration nodes pair
            if (u_ns_len == 0) or (v_ns_len== 0):
                # if a node has no predecessors then setting similarity to zero
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                print('\n s_uv=', s_uv)
                print('  u_ns_len *v_ns_len=', u_ns_len *v_ns_len)
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / ( u_ns_len *v_ns_len )
    return sim