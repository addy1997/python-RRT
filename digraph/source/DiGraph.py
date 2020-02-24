#!/usr/bin/env python
# coding: utf-8

# In[12]:


from Edge import *
from math import sqrt
from collections import defaultdict

class Digraph:

    def __init__(self):

        self.vertices   = []
        self.size       = len(self.vertices)
        self.edge_count = len(self.edges)
        self.edges      = []
        self.neighbour = defaultdict(set)

    def get_vertices(self):
        return self.vertices

    def get_edges(self):
        if (self.edges == None):
            raise ValueError("edge can't be zero or null")
        else:
            return self.edges

    def add_vertices(self, vertices):
        if (self.vertices is not None):
            self.vertices = []
        else:
            raise IOError("Node list is empty")
        

    def add_node(self,label,num=0):
        for v in self.vertices:
            if label == v.get_label():
                return v

            if label not in self.vertices:
                self.vertices.append(label)
                self.isolated.append(label)
                self.size += 1
            else:
                raise DeprecationWarning("the Node already exists")

    def add_edge(self, from_node, to_node, num=0):


        if (self.edges is not None):
            self.edges = []
        else:
            raise IOError("Edge list is empty")

        if from_node in self.vertices & to_node in self.vertices:
            E1 = Edge(from_node, to_node, num=num)
            E2 = Edge(from_node, to_node, num=num)
            if (from_node in self.isolated)&(to_node in self.isolated):
                self.isolated.remove(from_node)
                self.isolated.remove(to_node)

            if (len(E1) is None or len(E1)==0)&(len(E2) is None or len(E2)==0):

                raise ValueError("the length can't be None or 0")


    def dist(p1, p2):
        return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def isolated(self, maxD=1e309):
        """
        to locate the unvisited nodes to isolate.
        """
        visited = self.vertices
        preceding_node = {}
        T_distance = defaultdict(lambda: 1e309)
        self.isolated = isolated

        while visited:
            current = visited.intersection(T_dist.keys())
            if not current: break
            min_dist_node = min(current, key=T_distance.get)
            visited.remove(min_dist_node)

            for neighbour in self.neigbour[min_dist_node]:
                d = T_distance[min_dist_node] + self.dist[min_dist_node, neighbour]
                if T_distance[neigbour] > d and maxD >= d:
                    T_distance[neighbour] = d
                    preceding_node[neighbour] = min_dist_node
                    isolated = preceding_node[neighbour]

            return distance, preceding_node, isolated


    def Adj_matrix(self):

        encoding = enumerate([v for v in self.vertices])
        dim = len(self.vertices)
        adj = [[0] * dim] * dim

        for key, value in encoding:

            for i in range(dim):
                pass


    def to_dot_format(self, dotfile):

        dot_file = open(dotfile, "w")
        dot_file.write(Digraph {"\n"})
        for i in self.edges:

            dot_file.writelines(i.get_from_node_list() + "->" + i.get_to_node_list() + "\n")
            for j in self.isolated:
                dot_file.writelines(j + "\n")

            dot_file.write("}")
            dot_file.close()


    def is_connected(self):
        return (len(self.isolated)==0)


# In[ ]:
