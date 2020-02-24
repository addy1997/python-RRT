#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 

class Edge:
    """
    Edge 
    
    attr:
    -------------------
    node1(object) -> from_node -> source_node
    node1(object) -> to_node   -> target_node
    num_edges(int) -> capacity of edges 
    """
    
    def __init__(self, from_node, to_node, num):
        
        self.from_node = []
        self.source    = []
        self.target    = []
        self.to_node   = []
        self.cost      = 0
        self.num       = None

        self.length    = len(self.edge)
        
    def get_from_node_list(self):
        
        """
        use pandas to fetch the list
        """
        return self.from_node
    
    def get_to_node_list(self):
        
        """
        use pandas to fetch the list
        """
        return self.to_node
    
    def add_node(self, nodes, souce = True, target = True):
        
        nodes = []
        
        for m in nodes:
            for nodes in self.from_node:
                if (nodes != None & nodes != self.source & nodes!= self.target):
                    source.append(self.from_node[nodes])
                else:
                    raise ValueError("A vertex can't be Null")
                    
        for n in nodes:
            for nodes in self.to_node:
                if (nodes != None & nodes != self.source & nodes!= self.target):
                    target.append(self.to_node[node])
                else:
                    raise ValueError("A vertex can't be Null")
                    
        return source, nodes, target         
    
    def update_edge(self, edge):
        
        if (source != None & target != None):  
            self.from_node = self.source
            self.to_node   = self.target
        else:
            raise ValueError("Vertex can't be Null")
                             
        self.edge = [self.source, self.target]
        return self.edge 


# In[ ]:




