import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

n = 5

G=nx.gnp_random_graph(n,0.5,directed=True)
pos=nx.spring_layout(G)
G_n = nx.DiGraph()
Eps=np.zeros(shape=(n,n))
for i in range(n):
    for j in range(n):
        if G.has_edge(j,i):
            if G.predecessors(i)==1:
                Eps[j,i]=1
            else:
                if  j==max(G.predecessors(i)):
                    Eps[j,i]=1-Eps[:,i].sum()    
                else:
                    Eps[j,i]=(1-Eps[:,i].sum())*np.random.random_sample()            
print(Eps)


nx.draw(G_n,pos,node_size=100,label=range(n))
nx.draw_networkx_nodes(G_n,pos,node_size=140)
nx.draw_networkx_labels(G_n,pos,node_size=120)
plt.show()