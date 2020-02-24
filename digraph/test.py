#!/usr/bin/env python
# coding: utf-8

# In[5]:


from digraph_generator import *
from DiGraph import Digraph


def test(graph):
    vertices = specify_vertices(graph)
    edges    = specify_edges(graph)
    Digraph  = Digraph(vertices, edges)
    print (Digraph)
    
    Digraph.Adj_matrix()
    sp = Digraph.dist("a", "d")
    isolated = Digraph.isolated(maxD=1e309)
    connected = Digraph.is_connected()
    print(sp,"\n", isolated, "\n" ,connected)
    
    
def test_design(graph):
    G = design_graph_object(graph, Digraph())
    sp = G.dist("a", "d")
    isolated = G.isolated(maxD=1e309)
    connected = G.is_connected()
    
    return sp, isolated, connected
    
    
    


# In[ ]:




