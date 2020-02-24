#!/usr/bin/env python
# coding: utf-8

# In[2]:


from time import time
import networkx as nx
from source.DiGraph import DiGraph
from source.model.Edge import Edge 

def text_to_dict(filename):
    
    in_file = open("filename", "r")
    lines   = in_file.read()
    in_file.close()
    open_bracket = lines.index("{")
    close_bracket = lines.index("}")
    
    graph = eval(lines[open_bracket:close_bracket])
    return graph

def specify_vertices(graph):
    vertices = []
    for node in graph.keys():
        vertices.append(node)
    return vertices

def specify_edges(graph):
    edges = []
    for node in graph.key():
        edges.append(Edge(node, i))
    return edges

def design_graph_object(graph, G= None):
    
    if not G:
        G = DiGraph()
    for node in graph.keys():
        if (node not in G.get_vertices()):
            G.add_node(node)
        for z in graph[node]:
            if (z not in G.get_vertices()):
                G.add_node(z)
                G.add_edge(node, z)
    return G    


def set_digraph_library(graph, G):
    
    for nodes in graph.keys():
        G.add_node(nodes)
        for i in graph[nodes]:
            G.add_edge(nodes, i)
    return G
                
    
        


# In[ ]:




