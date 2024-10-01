from pygsp import graphs
import pygsp
from graph_coarsening.coarsening_utils import *
import networkx as nx
pygsp.plotting.BACKEND = 'matplotlib'


G = graphs.BarabasiAlbert(400)
if not hasattr(G, 'coords'): 
    try:
        graph = nx.from_scipy_sparse_array(G.W)
        pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')  
        G.set_coordinates(np.array(list(pos.values()))) 
    except ImportError:
        G.set_coordinates()
    G.compute_fourier_basis() # this is for convenience (not really needed by coarsening)
    
C, Gc, Call, Gall  = coarsen(G, r=0.5, method='variation_neighborhoods')

fig = plot_coarsening(Gall, Call)
plt.show()
print(Gall[0].info)