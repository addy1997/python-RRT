#!/usr/bin/env python
# coding: utf-8

# In[1]:


import test as t
from src.digraph_generator import text_to_dict

def main():
    """
    Main function of Algorithms for transmitting to test module
    while giving the filename,not necessary to add cd up symbol(..) at the beginning of file
    """
    filename = "data/graph.txt"
    dict_graph = text_to_dict(filename)
    t.test_digraph(dict_graph)
    t.test_design(dict_graph)

if __name__ == '__main__':
    main()


# In[ ]:




