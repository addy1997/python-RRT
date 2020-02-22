#!/usr/bin/env python
# coding: utf-8

# In[10]:


import copy 
import logging
import os
import torch
#from utils import start_idx_from_lengths


class Trajectories:
    
    def __init__(self, weights : torch.Tensor
        ,indices : torch.Tensor, 
        num_nodes : int, length : torch.Tensor,
        traversed_edges : torch.Tensor = None,
        pairwise_node_distances : torch.Tensor = None):
        
        """
        weights : w : parameter_weights - each row shoyuld sum up to 1
        -> number_observations * K

        lengths : L : parameter_lengths - contains number of observations in each trajectory or 
        the number of nodes the edge comprises.
        -> number_trajectories

        traversed/ visited edges 
        -> (total_edges - number_trajectory) or (number_observations- num_trajectories) * max_path_length 

        """
        
        assert weights.shape == indices.shape
        assert weights.dim() == 2
        
        
        #arguments 
        
        self.weights = weights
        self.indices = indices
        self.lengths  = lengths
        self.traversed_edges = traversed_edges
        self.starts = starts
        self.mask = None
        self.num_nodes = num_nodes
        self.pairwise_node_distances = pairwise_node_distances
        
        self.device = weights.device
        self.check_device()
        
        self.index_mapping = None
        self.num_trajectories = None
        
        if traversed_edges is not None:
            assert traversed_edges.shape[0] == self.lengths.sum() - len(self)
        
        
    def _len(self) -> int:
        if self.mask is None:
            self.num_trajectories = len(self.lengths)
        else:
            self.num_trajectories = self.mask.long().sum().item()
        return self.num_trajectories
    
    def get_item(self, values):
        
        item   = self.mapped_index(item)
        start  = self.starts[item]
        length = self.lengths[items]
        observations = torch.zeros([length, self.num_nodes], device = self.device)
        row = torch.arange(length).unsqueeze(1).repeat(1, self.weights.shape[1])
        observations[row, self.indices[start : start + length]] = self.weights[start : start + length]
        
        return observations 
    
    #property
    
    def lengths(self):
        
        if self.mask is None:
            return self.lengths
        else:
            return self.lengths[self.mask]
            
    def mapped_index(self):
        if self.mask is None:
            
            return index
        else:
            if self.index_mapping is None:
                self.index_mapping = self.mask.nonzero()[:,0]        
            return self.index_mapping[index]
        
        
    def traveresed_edges_by_trajectory(self, trajectory_id : int) -> torch.Tensor:
        item = self.mapped_index(trajectory_id)
        start = self.start[item] - item
        length = self.lengths[item] - 1
        traversed_edges = self.traversed_edges[start : start + length]
        return traversed_edges
    
    def traversed_edges(self, trajectory_id, bounce=None):
        
        if self.traversed_edges is None:    
            return stop
        elif traversed_edges == self.traversed_edges_by_trajectory(trajectory_id):
            return traversed_edges 
        if bounce is not None:
            traversed_edges = traversed_edges[bounce]
            traversed_edges = traversed_edges.flatten()
        if traversed_edges !=-1:
            return traversed_edges[traversed_edges]
        
    def foot_length(self, trajectory_id):
        
        traversed_edges = self.traversed_edges_by_trajectory(trajectory_id)
        lengths = (traversed_edges !=-1).sum(dim=1)
        return lengths
    
    def shortest_foot_lengths(self, trajectory_id):
        
        observations = self[trajectory_id]
        num_bounce = self.lengths[trajectory_id] -1
        min_distances = torch.zeros(num_bounce, device = self.device, dtype = torch.long)
        for bounce in range(num_bounce):
            from_nodes = observation[bounce].nonzero().squeeze()
            to_nodes   = observations[bounce].nonzero.squeeze()
            all_distances = self.pairwise_node_distances[from_nodes, :][:, to_nodes]
            min_distance[bounce] = all_distances[all_distances >= 0].min()
            
        return min_distance
    
    def clone(self) -> "Trajectories":
        """shallow copy"""
        return copy.copy(self)
    
    def to(self, device: torch.device) -> "Trajectories":
        
        if self.device == device:
            return self
        
        moved_trajectories = self.clone()
        moved_trajectories.device = device 
        for attribute, value in moved_trajectories.__dict__.items():
            if hasattr(value, "device") and value.device != self.device:
                raise ValueError(f"Trajectories attribute '{attribute}' is on device '{value.device}' instead of '{self.device}'")
                
                
    def with_mask(self, mask):
        if mask is not None and mask.device != self.device:
            mask = mask.to(self.device)
            
    masked_trajectories = self.clone()
    masked_trajectories.mask = mask
    masked_trajectories.reset_to_default()
    
    def reset_to_default(self):
        self.num_trajectories = None
        self.index_mapping = None
        
"""-------READ/ WRITE FILE--------"""

def write_to_directory(self, directory):
        if self.mask is not None:
            logging.warning("Trajectories mask ignored when writing to dictionary")
        
        os.makedirs(directory, exist_ok = True)
        
        with_open(os.path.join(directory, "lengths.txt"), "w") as f:
            for i, l in enumerate(self.lengths):
                f.write("{}\t{}\n".format(i, l.item()))
                
        with_open(os.path.join(directory, "observations.txt"), "w") as f:
                f.write("{}\t{}\n".format(*self.indices.shape))
                for row in range(self.indices.shape[0]):
                    row_elements = []
                    for col in range(self.indices.shape[1]):
                        row_elements.append(self.indices[row, col])
                        row_elements.append(self.weights[row, col])
                    f.write("\t".join(map(lambda x: str(x.item()), row_elements)))

        with_open(os.path.join(directory, "paths.txt"), "w") as f:
            f.write("{}\t{}\n".format(self.traversed_edges.shape))
            for foot in self.traversed_edges:
                line = "t".join(str(p.item()) for p in foot if p != -1) + "\n"
                f.write(line)
                
def read_from_files(
        cls, lengths_filename, observations_filename, paths_filename, num_nodes
    ):
        """
        Read trajectories from files `lengths.txt` `observations.txt` and `paths.txt`

        Length file has per line trajectory id and length Example
        ```
        0	9
        1	9
        2	7
        3	8
        4	7
        5	7
        ```

        Observations file start with num_observations, k (point per observation)
        then per line node_id, weight x k. Example:
        ```
        2518	5
        17025	0.22376753215971462	17026	0.2186635904321353	1137	0.18742442008753432	6888	0.20024607632540276	4585	0.16989838099521318
        6888	0.20106576291692577	1137	0.20348475328200213	4585	0.20255400616332436	1139	0.1985437138699239	6887	0.1943517637678238
        14928	0.18319982750248237	1302	0.18136407620166017	14929	0.1979849150163569	628	0.18905104643181994	1303	0.24840013484768056
        ```

        Paths file start with number of paths and maximum length
        Then per line, sequence of traversed edge ids. Example:
        ```
        2254	41
        20343	30411	30413	12311	1946
        1946	8179	30415	24401	24403	1957	8739	1960	24398	24400	20824	20822	20814	19664	19326	19327	26592	19346	29732	26594	13778	20817	13785	26595	26597
        ```
        """

        # read trajectories lengths
        with open(lengths_filename) as f:
            lengths = [int(line.split("\t")[1]) for line in f.readlines()]
            lengths = torch.tensor(lengths)

        # read observations, assume fixed number of observations
        obs_weights, obs_indices = None, None
        with open(observations_filename) as f:
            num_observations, k = map(int, f.readline().split("\t"))
            obs_weights = torch.zeros(num_observations, k)
            obs_indices = torch.zeros(num_observations, k, dtype=torch.long)

            for i, line in enumerate(f.readlines()):
                elements = line.split("\t")
                for n in range(k):
                    obs_indices[i, n] = int(elements[2 * n])
                    obs_weights[i, n] = float(elements[2 * n + 1])

        # read underlying paths
        paths = None
        if paths_filename is not None and os.path.exists(paths_filename):
            with open(paths_filename) as f:
                num_paths, max_path_length = map(int, f.readline().split("\t"))
                paths = torch.zeros([num_paths, max_path_length], dtype=torch.long) - 1
                for i, line in enumerate(f.readlines()):
                    ids = list(map(int, line.split("\t")))
                    if len(ids) == 0:
                        print(i)
                    paths[i, : len(ids)] = torch.tensor(ids)

        return Trajectories(
            weights=obs_weights,
            indices=obs_indices,
            num_nodes=num_nodes,
            lengths=lengths,
            traversed_edges=paths,
        )
                


# In[ ]:




