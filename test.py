from graph_coarsening.coarsening_utils import *
import graph_coarsening.graph_utils

import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx
import pygsp as gsp
from pygsp import graphs
gsp.plotting.BACKEND = 'matplotlib'

N = 400

G = graphs.BarabasiAlbert(N)
if not hasattr(G, 'coords'): 
    try:
        graph = nx.from_scipy_sparse_array(G.W)
        pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')  
        G.set_coordinates(np.array(list(pos.values()))) 
    except ImportError:
        G.set_coordinates()
    G.compute_fourier_basis() # this is for convenience (not really needed by coarsening)
    
N = G.N
L = G.L.toarray()
S = graph_coarsening.graph_utils.get_S(G).T

plt.spy(G.W, markersize=0.2);
plt.savefig('barabasi_albert.png')

method = 'variation_neighborhood'  

# Parameters
r    = 0.6 # the extend of dimensionality reduction (r=0 means no reduction)
k    = 5  
kmax = int(3*k)
        
C, Gc, Call, Gall = coarsen(G, K=k, r=r, method=method) 
metrics = coarsening_quality(G, C, kmax=kmax)
n = Gc.N

print('{:16} | r: {:1.4}, nedges: {}, levels: {}, epsilon: {:1.4}'.format(method, metrics['r'], metrics['m'], len(Call), metrics['error_subspace'][k-1]))