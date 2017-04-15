import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# Generates random left stochastic matrix such that the 
# principal diagonal entries are zero
def gen_random_ls_matrix(N):
    A = np.zeros((N, N))
    for i in xrange(N):
        r = np.random.uniform(size=N-2)
        r = np.sort(r)
        diffs = np.ediff1d(r)
        d0, dn = r[0], 1.0 - r[-1]
        if i == 0:
            A[1][0], A[N-1][0] = d0, dn
            A[2:-1, 0] = diffs
        elif i == N-1:
            A[0][N-1], A[N-2][N-1] = d0, dn
            A[1:N-2, N-1] = diffs
        else:
            A[0][i], A[N-1][i] = d0, dn
            A[1:i, i] = diffs[:i-1]
            A[i+1:N-1, i] = diffs[i-1:]

    for i in xrange(N):
        for j in xrange(1, N):
            A[i][j] += A[i][j-1]

    return A


# Generates random graph with random weights
def get_incidence_matrix(N, p = 0.5):
    G =  nx.gnp_random_graph(N, p, directed = True)
    A = np.zeros((N,N))
    for i in range(N):
        m = max(G.predecessors(i))
        for j in range(N):
            if G.has_edge(j,i):
                if G.predecessors(i) == 1:
                    A[j,i] = 1
                else:
                    if j == m:
                        A[j,i] = 1.0 - A[:,i].sum()    
                    else:
                        A[j,i] =(1.0 - A[:,i].sum()) * np.random.random_sample()  

    for i in xrange(N):
        for j in xrange(1, N):
            A[i][j] += A[i][j-1]

    return A

