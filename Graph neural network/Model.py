#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from typing import Optional
from torch.nn import Tanh

#from graph import Graph


# In[4]:


class MLP(nn.Module):
    def __init__(self, input, f_in : int, f_out : int):
        
        """
        A Multilayer perceptron with 2 fully convolutional layers and 
        an input and output layer.
        
        Mathematically, given by the equation
        
        F(x) = G(b^(2) + W^(2)(s(b^(1) + w^(1)*(x))))  .......eq(1)
        or 
        
        F(x) = G(b^(2) + w^(2)(H(x))   ..........eq(2)
        
        where H(x) = s(b^(1) + w^(1)*x) ........eq(3)

        here 
        - G & s are the Activation functions such as Tanh or sigmoid
        - b^(2), b^(1) are biases and W^(1), W^(2) are weight matrices. 
        - F(x) is the output for the neural net.
                
        Args
        ------------------------------------------------------------
        
        1) input - a matrix or a tensor 
        2) f_in - dimension of the input which determins the range of the 
        activation function.
        3) f_out- dimension of the output which determins the range of the 
        activation function.
        """
         
        
        super(self, MLP).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.input = input
        self.fc_layer_1 = nn.Linear(f_in, f_in*4)
        self.fc_layer_2 = nn.Linear(f_in*4, f_out)
        
        
    def forward(self):
        return self.fc_2_layer(torch.nn.Tanh(self.fc_1_layer(X)))
    
    
class Model(nn.Module):
    
    def __init__(self,
        diffusion_graph_transformer: Optional["EdgeTransformer"],
        direction_edge_mlp: Optional[MLP],
        number_observations: int,
        rw_expected_steps: int,
        rw_non_backtracking: bool,
        latent_transformer_see_target: bool,
        double_way_diffusion: bool,
        diffusion_self_loops: bool,
    ):
        
        """
        shapes:
        
        observations: [traj_length, n_node]
        graph : Graph
        observed : [n_pred, numb_obs]
        starts : [n_pred, ] start indices in observations
        target : [n_pred, ] target indices in observations
        pairwise_node_features : [n_node, n_node]
        steps : [traj_length -1, ] distance b/w consecutive observations
        """
        
        """
        diff_graph_transformer : for counting the edge weight and improve graph learning
        
        multi_channel_diff : computes the diffusion of the past observation on the graph
        
        direction_edge_mlp : assigning directions to the graph edges and calculating the latent edge weights
        
        number_obs : number of observations
        
        rw_expected_steps: num of steps to be taken by the path generator
        
        rw_non_backtracking : the edges from where backtracking is impossible
        
        latent_transformer_see_target : show node cross features of target as input to direction_edge_mlp
            
        double_way_diffusion : multichannel_diffusion is run on the graph and reversed graph (reversed edge direction)
            
        diffusion_self_loops (bool): add self loop edges to all nodes on the diffusion graph

        """
        
        super(Model, self).__init__()

        # params
        self.number_observations = number_observations
        self.rw_expected_steps = rw_expected_steps
        self.rw_non_backtracking = rw_non_backtracking
        self.latent_transformer_see_target = latent_transformer_see_target
        self.double_way_diffusion = double_way_diffusion
        self.diffusion_self_loops = diffusion_self_loops

        # modules
        self.diffusion_graph_transformer = diffusion_graph_transformer
        self.direction_edge_mlp = direction_edge_mlp

    def forward(
        self,
        observations,
        graph: Graph,
        diffusion_graph: Graph,
        observed,
        starts,
        targets,
        pairwise_node_features,
        steps=None,
    ):
        
        assert observed.shape[0] == starts.shape[0] == targets.shape[0]
        n_pred = observed.shape[0]
        
        if self.diffusion_graph_transformer is None and self.direction_edge_mpl is None:
            
            if graph.edges.shape != torch.size([graph.n_edge, 1]) or ((graph.Out_nodes_counts -1.0).abs() > 1e-5).any():
                rw_graphs = graph.update(edges=torch.ones([graph.n_edge, n_pred], device = graph.device))
                
                rw_graphs = rw_graphs.c_weights()
        else:
                rw_graphs = graph
                virtual_coords = None
                virtual_coords = self.compute_diffusion(diffusion_graph, observations)
                
        if self.double_way_diffusion:
                
                virtual_coords_reversed = self.compute_diffusion(diffusion_graph.reversed_edges(), observations)
                    
                virtual_coords = torch.cat([virtual_coords, virtual_coord_reversed])

                rw_graphs - self.compute_rw_weights(virtual_coords, observed, pairwise_node_features, targets, graph)
                
                target_distributions = self.compute_random_walk(rw_graphs, observations, starts, targets, steps)
            
                return target_distributions, virtual_coords, rw_weights
        
        
                
    def compute_diffusion(self, graph, observations) -> torch.Tensor:
        
        if self.diffusion_self_loops:
            graph = graph.add_self_loops()
            
        if self.diffusion_graph_transformer:
            diffusion_graph = self.diffusion_graph-transformer(graph)
            
        else:
            
            diffusion_graph = graph.update(edges= torch.ones([graph.n_edge, 1], device = graph.device))
        
        #computing the softmax weights
        diffusion_graph = diffusion_graph.c_weights()
        
        #computing the weights for every observations
        diffusion_graph = diffusion_graph.update(nodes= observation.t())
        
        #compute the virtual coordinates for the diffusion_graph
        virtual_coords = self.multichannel_diffusion(diffusion_graph)
        
        return virtual_coords
    
    
    def compute_rw_weights(self, virtual_coords, observed, 
        pairwise_node_fetaures, targets, graph : Graph) -> Graph:
        
        n_pred = observed.shape[0]
        witness_features = []
        
        diffusions = virtual_coords[:, observed].view(graph.n_node, n_pred, -1)
        witness_features.append(diffusions[graphs.senders])
        witness_features.append(diffusions[graphs.receivers])
        
        
        #original node features
        if graph.nodes is not None:
            nodes = graph.nodes.view(graph.n_node, 1, -1).repeat(1, n_pred, 1)
            witness_features.append(nodes[graph.senders])
            witness_features.append(nodes[graph.receivers])
            
        #pairwise_node_features
        if self.latent_transformer_see_target and pairwise_node_features is not None:
            target_features = pairwise_node_features[targets].transpose(0, 1)
            witness_features.append(target_features[graph.senders])
            witness_features.append(target_features[graph.receivers])
        
        if graph.edges is not None:
            witness_featuresa.append(graph.edges.view(graph.n_edge, 1, -1).repeat(1, n_pred, 1))
        
        if self.latent_transformer_see_target and pairwise_node_features is not None:
            witness_features.append(graph.nodes[targets].unsqueeze(0).repeat(graph.n_edge, 1, 1))
            edge_input = torch.cat(witness_features, dim=2)
            edge_input = edge_input.view(n_pred * graph.n_edge, -1)
            rw_weights = self.direction_edge_mlp(edge_input).view(graph.n_edge, -1)
            
            rw_graphs = graph.update(edges=rw_weights)
            rw_graphs = rw_graphs.c_weights()
            
            return rw_graphs
        
        
    def compute_random_walk(self, rw_graph, observations, starts, targets, steps) -> torch.Tensor:
            
        n_pred = len(starts)
        n_node = observations.shape[1]
        device = observations.device
        rw_weights = rw_graph.edges.transpose(0, 1)
            
        start_distributions = observations[starts]
        rw_steps = self.computer_steps(starts, targets, steps)
        
        predict_distributions = torch.zeros(n_pred, n_node, device = device)
        
        for pred_id in range(n_pred):
            rw_graph = rw_graphs.update(edges = rw_weights[pred_id])
            
            max_step_rw = None
        if self.rw_expected_steps:
            max_step_rw = rw_steps[pred_id]
            start_nodes = start_distributions[pred_id]
            
        if self.rw_non_backtracking:
            predict_distributions[pred_id] = rw_graph.non_backtracking_random_walk(start_nodes, max_step_rw)
        else:
            predict_distributions[pred_id] = rw_graph.random_walk(start_nodes, max_step_rw)
            
            return predict_distributions 
        
    def compute_steps(starts, targets, steps):
        if steps is None:
            return None
        
        compute_steps= torch.cat([0], torch.cumsum(steps, dim=0), device = steps.device)
        
        return compute_steps[targets] - compute_steps[starts]
    

class Edge_Transformer(nn.Module):
    def __init__(self, d_node : int, d_edge : int, d_edge_out : int, activation = torch.sigmoid):
        
        super(Edge_Transformer, self).__init__() 
        d = 2* d_node + d_edge
        self.fc_layer_1 =  nn.Linear(d, 2*d)
        self.fc_layer_2 =  nn.Linear(2*d, d_edge_out)
        self.activation_func = activation
        
        
    def forward(self, graph):
        in_features = []
        if graph.nodes is not None:
            nodes = graph.nodes.view(graph.n_node, -1)
            in_features.append(nodes[graph.senders])
            in_features.append(nodes[graph.receivers])
            
        if graph.edges is not None:
            edges = graph.edges.view(graph.n_edge, -1)
            in_features.append(edges)
            
            in_features = torch.cat(in_features, dim =1)
            new_edges - self.fc_layer_2(self.activation(self.fc_layer_1(in_features)))
            return graph.update(edges=new_edges)            

