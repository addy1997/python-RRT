from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np #a library to play with arrays, matrices etc.
from math import sqrt
import pygsp #a signal processing library
import scipy #for mathematical calculations
import torch #framework
from typing import Union, List
from torch_scatter import scatter_add, scatter_max
import matplotlib as mpl #graphics library
from sklearn.utils import graph_shortest_path #python-based library for statistics and graphics
#import DiGraph
from collections import defaultdict

print("reference for the code - https://pytorch.org/docs/stable/torch.html")


import random, time
import torch
import datetime
import io
import array, re, itertools
import networkx as nx
from itertools import groupby




########### GRAPH PROPERTIES AND BASIC IMPLEMENTATION ###########
class Edge(object):

    def __init__(self):

        self.from_node = []
        self.to_node = []
        self.edges = sqrt((self.from_node[2] - self.to_node[2] )**2 + (self.from_node[3]- self.to_node[3])**2)

    def calculate_edge_length(self):
        length = self.edges
        if length == 0 or length == float('inf'):

            raise ValueError("The length can't be 0 or infinite")
        else:
            return length

#edge = Edge()

#edge.calculate_edge_length()


class Graph(Edge):

    def __init__(self):

        self.senders = senders
        self.receivers = receivers
        self.nodes = nodes
        self.edges = edges
        self.n_node = n_node or nodes.shape[0]
        self.n_edge = n_edge or edges.shape[0]
        self.device = senders.device
        self.pygsp = None
        self.coordinates = None
        self.pairwise_distance_chunk = None

        self.non_backtracking_edge_senders
        self.non_backtracking_edge_random_walk_graph
        self.non_backtracking_edge_receivers

        self.check_shapes()
        self.check_device = torch.device()

    def In_nodes_counts(self) -> torch.Tensor:
        """
        Compute the number of incoming nodes
        """
        shape = [self.n_node, *self.edges.shape[1:]]
        weights = torch.zeros(shape, device=self.device)
        scatter_add(src=self.edges, index = self.receivers, out=weights, dim=0)
        return weights


    def Out_nodes_counts(self) -> torch.Tensor:
        """
        Compute the number of outgoing nodes
        """
        shape = [self.n_nodes, *self.esges.shape[1:]]
        weights = torch.zeros(shape, device=self.senders.device)
        scatter_add(src=self.edges, index = self.senders, out = weights)
        return weights

    def In_edges_counts(self):

        """
        computes the number of in coming edges per node
        returns torch.LongTensor : [new_node,] per node
        """
        return scatter_add(src=torch.ones(self.receivers.shape,
        dtype = torch.long, device=self.device,
        index=self.receivers, dim=0, dim_size=self.n_node))


    def out_edges_counts(self) -> torch.Tensor:
        """
        computes and returns the count of the out going edges per node
        return torch.tensor : [new_node, ] per node
        """

        return scatter_add(src=torch.ones(self.senders.shape,
        dtype=torch.long, device=self.device), index = self.senders,
        dim=0, dim_size=self.n_nodes)

    def calculate_pairwise_distance(self) -> torch.Tensor:
        if self.pairwise_distance is None:

            G = DiGraph()
            G.add_edges(zip(numpify(self.senders), numpify(self.receivers)))
            G.add_nodes(range(self.n_nodes))

            self.pairwise_distance = torch.zeros([self.n_node, self.n_node], device = self.device)-1

            for source, targets in graph_shortest_path(G):
                for target, length in targets.items():
                    self.pairwise_distance[source, target] = length

        return self.pairwise_distance


    def Edge(self, sender: int, receiver : int):

        and_gate = (self.senders==sender) & (self.receivers==receiver)
        edge = self.edges[and_gate].squeeze()
        return edge

    def Dense_Adj_Matrix(self) -> torch.Tensor:

        edges = self.edges.squeeze()
        transition_matrix = torch.zeros([self.n_node, self.n_node],device = self.device)

        transition_matrix[self.senders, self.receivers] = edges
        return transition_matrix

    def coordinates(self) -> torch.Tensor:

        coords = []
        if self.coords is None:
            self.coords_from_features()

        return self.coords


    def compute_edge_vectors(self):
        vector = self.coordinates[self.receivers]-self.coordinates[senders]
        return vector

    def max_edge_weight_per_node(self) -> torch.Tensor:

        return scatter_max(self.edges.squeeze(), self.senders)

    def max_edge_vector_per_node(self) -> torch.Tensor:

        c_weights, edge_ids = self.max_edge_weight_per_node()
        c_weights = c_weights - (1 / self.out_edge_counts.float())
        c_weights[self.out_edge_counts.float() == 0] =0
        return self.compute_edge_vectors()[edge_ids] * weights.unsqueze()


    def reverse_edges(self) -> 'Graph':
        """

        return a reversed graph
        """

        return self.update(senders = self.receivers, receivers = self.senders)


    def reorder_edges(self):

        """
        returns a sorted list of edges or graph. A general sorting mechanism is used based on the indices.
        """

        senders = numpify(self.senders)
        receivers = numpipy(self.recivers)

        indices = np.agsort(receivers)
        nxt_indices = np.agsort(senders[indices], kind = 'radixsort')
        new_indices = indices[nxt_indices]
        new_indices = torch.tensor(new_indices, device = self.device)


        #quantised_c = torch.quantize_per_channel()
        #quantised_t = torch.quantize_per_tensor()

        return self.update(senders=self.senders[new_indices], receivers=self.receivers[new_indices],
                          edges=self.edges[new_indices])

    def avoid_self_loops(self) -> 'Graph':

        mask = self.senders == receivers
        return self.update(senders = self.senders[~mask], receivers= self.receivers[~mask], edges=self.edges[~mask], new_edge = (~mask).long().sum())


    def __matmul__(self, node_signal: torch.Tensor) -> torch.Tensor:

        """
        product = input * weight
                = node_signal * W
        """
        assert node_signal.shape[0] == self.n_node
        assert self.edges is not None and self.edges.squeeze().dim()==1
        senders_features = node_signal[self.senders]
        broadcast_edges = self.edges.view(-1, *([1]* (node_signal.dim() -1)))
        weighted_senders = senders_feaures * broadcast_edges
        node_results = scatter_add(src= weighted_senders, index = self.receivers, dim=0, dim_size= self.n_node)
        return node_results


    def add_self_loops(self, edge_value: float=1, degree_zero_only =False):

        if degree_zero_only:
            add_self_loops_nodes = (self.out_degree_counts == 0).nonzero()[:, 0]
        else:
            add_self_loop_nodes = torch.arange(self.n_node, device = self.device)

        new_senders = torch.cat([self.senders,add_self_loop_nodes])
        new_receivers = torch.cat([self.receivers, add_self_loop_nodes])
        new_edges = torch.cat([self.edges, edge_value*torch.ones([len(add_self_loop_nodes), *self_edges.shape[1:]], device = self.device)])

        return self.update(senders= new_senders, receivers = new_receivers, edges= new_edges, new_edge = self.n_edge + len(add_self_loop_nodes))

    def normalize_weight(self) -> 'Graph':

        new_edges = self.edges/ self.out_degree[self.senders]
        return self.update(edges=new_edges)

    def softmax_weights(self):

        max_out_weights_per_node, _ = scatter_max(src=self.edges, index= self.senders, dim=0, dim_size= self.n_node, fill_value = 1e20)
        shifted_weights = self.edges = mac_out_weight_per_node[self.senders]

        exp_weights = shifted_weights.exp()
        normalizer = scatter_add(src= exp_weights, index= self.senders, dim=0, dim_size = self.n_node)
        sender_normalizer = normalizer[self.senders]
        normalized_weights = exp_weights/ sender_normalizer

        if is_nan(normalized_weights):
            logging.warning("NaN weight after normalization in graph 'softmax_weights'")

        return self.update(edges=normalized_weights)

    def extract_coords_from_features(self, keep_in_features: bool = True):

        assert self.nodes is not None and self.nodes.shape[1]>=2

        self.coordinates = self.nodes[:, :2]

        if not keep_in_features:
            if self.nodes.shape[1] == 2:
                self.nodes = None
            else:
                self.nodes = self.nodes[:, 2:]

    def edge_features_with_nodes(self) -> torch.Tensor:

        if self.augmented_edge_features is None:
            features = []
            if self.edges is not None:
                features.append(self.edges)
            if self.nodes is not None:
                features.append(self.nodes[self.senders])
            if self.nodes is not None:
                features.append(self.nodes[self.receivers])
            self.augemented_edge_features = torch.cat([f.view(self.n_edge, -1) for f in features], dim=-1)

        return self.augmented_edge_features


############ RANDOM WALKS #############
#In this section we use different sampling techniques for sampling the graph


    def random_walk_sampling(self, star_node: Union[int, torch.Tensor],
    samples:int, steps:int, allow_to_rollback : bool =True):

    """
    This function samples the graph
    Attr
    ----------------------------------------------------------
    star_node - The start node's index.
    samples   - a counter to keep track of the number samples.
    steps     - number of steps for each sample.
    allow_rollback - return back to the previous node/ state.

    """
    start_nodes = torch.zeros(samples, device = self.device, dtype=torch.long)
    if type(start_node) is int or (type(start_node) is torch.Tensor and start_node.dim() ==0):
        start_nodes[:] = start_node
    else:
        for i in range(samples):
            start_node[i] = sample(torch.arange(len(start_node)), start_node)



    visited_nodes = torch.zeros([samples, steps+1], device = self.device, dtype=torch.long)-1
    visited_nodes[:, 0] = start_nodes
    visited_nodes = torch.zeros([samples, steps], device = self.device, dtype= torch.long) -1

    for i_th_sample in range(samples):
        current_node = start_nodes[i_th_sample]
        for step in range(steps):
            possible_edges_mask = self.senders == current_node
            if not allow_rollback and step>=1:
                possible_edges_mask = (self.receivers != visited_nodes[i_th_sample, step -1])


    edge_ids = possible_edges_mask.nonzero()[:, 0]
    occcupied_edge = sample(edge_ids, self.edges[edge_ids] / self.edges[edge_ids].sum())
    current_node = self.receivers[occupied_edge]

    visited_edge[i_th_sample, step] = occupied_edge
    visited_edge[i_th_sample, step+1] = cuurent_node

    unvisited_nodes = self.nodes - visited_nodes

    if unvisited_nodes == 0 :
        return visited_nodes, visited_edges, unvisited_edges


    def random_walk(self, start_nodes : torch.Tensor, steps : int) -> torch.Tensor:
        """
        Take a random walk around the graph to examine the probsbility distribution of the sampled nodes

        """
        if steps == 0:
            return start_nodes

        nodes_signal = start_nodes
        for _ in range(steps):
            nodes_signal = self @ nodes_signal
        return nodes_signal

    def compute_non_backtracking_edges(self) -> (torch.LongTensor, torch.LongTensor):

        """
        This function return the edges from among (self.senders and self.receivers )
        which have no possibility of backtracking.
        -----------------------------------------------------------------------------------

        Eg: The First/ Main nodes of the graph or the isolated nodes of the graph

        """
        if self.non_backtracking_edge_senders is None or self.non_backtracking_edge_receivers is None:
            senders, recivers = self.senders.to("cpu"), self.receivers.to("cpu")

            continuing_edges = receivers.unsqeeze(1) == senders.unsqueeze(0)
            looping_edges    = senders.unsqueeze(1) == receivers.unsqueeze(0)
            non_backtracking_edges = continuing_edges & ~looping_edges


            nbe = non_backtracking_edges.nonzero().to(self.device)
            edge_senders = nbe[:, 0]
            edge_receivers = nbe[:, 1]
            self.non_backtracking_edge_senders, self.non_bactracking_edge_receivers


        return self.non_backtracking_edge_senders, self.non_bactracking_edge_receivers


        def non_backtrackin_random_walk_graph(self) -> 'Graph':

            if self.non_backtracking_random_walk_graph is None:
                edge_senders, edge_receivers = self.compute_non_backtracking_edges()
                edge_weights = self.edges[self.receivers]
                G= Graph(senders = edge_senders,
                        receivers = edge_receivers,
                        nodes = None,
                        edges = edge_weights,
                        new_nodes = self.n_edges,
                        new_edges = len(edge_weights)

                G.add_self_loops(degree_zero_only)
                G.normalize_weights()
                self.non_backtracking_random_walk_graph = G

                return self.non_backtracking_random_walk_graph


        def non_backtracking_random_walk(self, start_nodes : torch.Tensor, steps : int) -> torch.Tensor

           if steps == 0:
              return start_nodes

            edge_start  = start_nodes[self.senders] * self.edges
            edge_signal = edge_start
            for _ in range(steps -1)

            edge_signal = self.non_backtracking_random_walk_graph @ edge_signal

            node_signal = scatter_add(src= edge_signal, index= self.receivers, dim_size=self.new_node)

            return node_signal


     ####plotting####

class Graph_plotting:
    def __init__(self):
        self.plot_signal = pygsp.plot_signal(*[numpify(a) for a in args], **{k: numpify(v) for k, v in kwargs.items()})
        self.plot        = pygsp.plot(*[numpify(a) for a in args], **{k: numpify(v) for k, v in kwargs.items()})



    def signal_plotting(self, *args, **kwargs):
        """
        calls pygsp.plot_signal function
        """
        return self.plot_signal

    def make_plot(self, *args, **kwargs):
        """
        calls the pygsp.plot function
        """
        return self.plot


   def plot_trajectory(self, distributions : torch.Tensor, color : list,
                        with_edge_arrows : bool= False,
                        highlight : Union[int, List[int]] = None,
                        zoomed : bool = False,
                        ax=None, normalize_intercept : bool = False,
                        edge_width : float = .1):

     if ax == None:
         fig = plt.figure()
         ax = fig.add_subplot(111)

     if zoomed:
         display_points_mask = distributions.sum(dim=0) > 1e-4
         display_coords      = self.coordinates[display_points_mask]
         x_min, x_max        = (display_coords[:, 0].min(), display_coords[: 0].max())
         y_min, y_max        = (display_coords[:, 1].min(), display_coords[:,1].max())
         x_center, y_center  = (x_min + x_max)/2 , (y_min + y_max)/2
         size                = max(x_max - x_min, y_max - y_min)
         margin              = size * 1.1 / 2
         ax.set_xlim([x_center-margin, x_center+margin])
         ax.set_ylim([y_center-margin, y_center+margin])

         #plot the edges
         vertex_size = 0.
         if highlight is not None:
             vertex_size = np.zeros(self.new_node)
             vertex_size[highlight]  = .5

        self.pygsp.plotting['highlight_color'] = gnx_plot.green
        self.pygsp.plotting['normalize_intercept'] = 0.
        self.plot(edge_width = edge_width,
                  edges = True,
                  vertex_size = vertex_size,
                  vertex_color = [(0. , 0., 0., 0.)] * self.new_node
                  highlight = highlight,
                  ax=ax)

        transparent_colors = [mpl.colors.to_hex(mpl.colors.to_rgba(c, alpha=0.5), keep_alpha=True) for c in colors]

        self.pygsp.plotting['normalize_intercept'] = 0

    for distribution, color in zip(distributions, transparent_colors):
        self.plot(vertex_size  = distribution,
                     vertex_color = color,
                      edge_width   = 0,
                      ax = ax)


    if with_edge_arrows:
        coordinates = self.coordinates
        arrows = self.max_edge_vector_per_node()
        coordinates = numpify(coordinates)
        arrows = nimpify(arrows)
        ax.quiver(coordinates[:, 0],
                      coordinates[:, 1],
                      coordinates[:, 0],
                      coordinates[:, 1],
                      pivot = 'tail')

    ax.set_aspect('equal')
    return ax


    ######READ/ WRITE/ INTO/ FROM THE FILES##########

    def pygsp(self) -> pygsp.graphs.Graph:

        """
        Function to create the a pygsp graph

        """
        if self.pygsp is None:
            weights = self.edges.squeeze()
            assert weights.dim() == 1
            weights = weights.detach().to('cpu').numpy()
            senders = self.senders.detach().to('cpu').numpy()
            receivers = self.receivers.detach().to('cpu').numpy()

            W = scipy.sparse.coo_matrix((weights, (senders, receivers)))
            coordinates = self.coordinates.detach().to('cpu').numpy()
            self.pygsp = pygsp.graphs.Graph(W, coordinates= coordinates)

    return self.pygsp

    def from_pygsp_graph(cls, G) -> 'Graph':
        """
        Construct a GRpah from a PYGSP graphs
        """

        senders, receivers, weights = map(torch.Tensor, G.get_edge_list())
        senders = senders.long()
        receivers = receivers.long()
        edges = weights.float()
        new_edges = G.new_edges

        #consider a directed graph

        if not G.is_directed():
            senders, receivers = torch.cat([senders, receivers]), torch.cat([senders, recivers])
            edges = torch.cat([edges, edges])
            new_edges *= 2

        nodes = torch.tensor(G.coordinates).float() if G.coordinates is not None else normalize_intercept

        return Graph(senders=senders,
                     receivers = receivers,
                     edges = edges,
                     nodes = nodes,
                     new_node = G.new_vertices,
                     new_edges = new_edges)


    ########READ/ WRITE#################
    def read_from_files(cls, nodes_filename: str, edges_filename: str) -> 'Graph':

        node_features = None
        edge_features = None

        #read node features
        with open(nodes_filename) as f:
            num_nodes, num_node_features = map(int, f.readline().split('\t'))
            if num_node_features > 0:
                node_features = torch.zeros(num_nodes, num_node_features)
                for i, line in enum(f.readlines()):
                    features = torch.tensor(list(map(float, line.split('\t')[1:])))
                    node_features[i] = features

        with open(edges_filename) as f:
            num_edges, num_edges_features = map(int, f.readline().split('\t'))

            senders = torch.zeros(num_edges, dtype=torch.long)
            receivers = torch.zeros(num_edges, dtype=torch.long)

        if num_edge_features > 0:
            edge_features = torch.zeros(num_edges, num_edge_features)

        for i, line in enum(f.readlines()):
            elements = line.aplit('\t')
            senders[i] = int(elements[1])
            receivers[i] = int(elements[2])
            if edge_features is not None:
                edge_features[i] = torch.tensor(list(map(float, elements[3:])))


        return Grpah(nodes = node_features, edges= edge_features, senders=senders, receivers=receievers, new_node = num_nodes, new_edge= num_edges)


    def write_to_directory(self, directory: str):
        os.makedirs(directory, exist_ok = True)

        with open(os.path.join(directory, 'nodes.txt'), 'w') as f:
            f.write("{}\t{}\n".format(self.n_node, 0 if self.nodes is None else self.nodes.shape[1]))
                if self.nodes is not None:

                    for i,features in enum(self.nodes):
                        line = str(i) + "\t" +  "\t".join(map(str, [f.item() for f in features])) + "\n"

                        f.write(line)

        edges = self.edges
        if edges is not None and edges.dim() == 1:
            edges = edges.unsqueeze(-1)
        if edges is None:
            edges = [[]] * self.n_edge

        with open(os.path.join(directory, 'edges.txt'), 'w') as f:
            f.write("{}\t{}\n".format(self.n_edge, 0 if self.edges is None else self.edges.shape[1]))
            for i , (sender, receiver, features) in enum(zip(self.senders, self.receivers, edges)):

            line = "\t".join(map(str, [i, sender.item(), receiver.item()])) + \
                   "\t" + \
                   "\t".join(map(str, [i, sender.item(), receiver.item()])) + "\n"
            f.write(line)


    def from_networkx_graph(graph, node_feature_field : str = 'feature',
                            edge_feature_field : str = 'feature') -> 'Graph':

    """
create graph from a networkx graph

    """

    net = nx.convert_node_labels_to_integers(graph)
    n_node = net.number_of_nodes
    n_edge = net.number_of_edges
    senders = torch.tensor([e[0] for e in net.edges()])
    receivers = torch.tensor([e[1] for e in net.edges()])

    ndoes = None

    if n_node > 0 and node_feature_field in net.nodes[0]:
        shape = [n_node, *torch.tensor(net.nodes[0][nodes_feature_field]).shape]
        if len(shape) == 1:
            shape = [shape[0] , 1]
        nodes = torch.zeros(shape).float()
        for i in range(n_node):
            nodes[i] = torch.tensor(g.nodes[i][node_feature_field])

    edges = None

    if n_edge > 0:
        first_edge_data = next(iter(net.edges(data=True)))[2]
        if edge_feature_field in first_edge_data:
            shape = [n_edge, * tensor.tensor(first_edge_data[edge_feature_field]).shape]

            if len(shape) == 1:
                shape = [shape[0], 1]
                edges = torch.zeros(shape).float()
                for i, (_,_, features) in enum(net.edges.data(edge_feature_field)):
                    edges[i] = torch.tensor(features)

    if not net.is_directed():
        if edges is of None:
            edges = torch.cat([edges, edges])
        senders, receivers = torch.cat([senders, receivers]), torch.cat([receivers, senders])

    net = Graph(nodes= nodes, dges = edges, receivers = receivers, senders= senders, n_ndoe =n_node, n_edge=n_edge)

    return net

    """ Graph semantics """

    def __repr__(self):

        nodes_str = None if self.nodes is None else list(self.nodes.shape)
        edges_str = None if self.edges is None esle list(self.edges.shape)

        return f"Graph(n_node = {self.n_node}, n_edge = {self.n_edge}, nodes= {nodes_str}, edges = {edges_str}"

    def clone(self) -> 'Graph':
        return copy.copy(self)

    def update(self, **kwargs):


        for k in kwargs.items():
            if k[0] == "":
                raise ValueError(f"Graph update should not affect _protected attribute '{k}'")

        net = self.clone()


        for k, v in kwargs.items():
            setattr(g, k, v)

        if any(k in ["receivers", "senders"] for k in kwargs):
            net.non_backtracking_edge_senders = None
            net.non_backtrackin_edge_receivers = None

        if any(k in ["receivers", "senders", "edges"] for k in kwargs):
            net.non_backtracking_random_walk_graph = None

        if any(k in ["receivers", "senders"] for k in kwargs):
            net.distances = None

        if any(k in ["receivers", "senders", "nodes", "edges"]
               for k in kwargs):
            net.pygsp = None

        g._check_device()
        g._check_shapes()
        return g

    def to(self, device: torch.device) -> 'Graph':
        """Move this Graph instance to the required device

        Returns:
            Graph: moved Graph
        """
        if self.device == device:
            return self

        moved_graph = self.clone()
        moved_graph.device = device
        for attribute, value in moved_graph.__dict__.items():
            if type(value) is torch.Tensor or type(value) is Graph:
                moved_graph.__dict__[attribute] = value.to(device)
        return moved_graph

    def check_device(self):
        """Check that all attributes of the graph are on `self.device`

        Raises:
            ValueError: if a tensor is not on the right device
        """
        for attribute, value in self.__dict__.items():
            if hasattr(value, 'device') and value.device != self.device:
                raise ValueError(
                    f"Graph attribute '{attribute}' is on device '{value.device}' instead of '{self.device}'"
                )

    def check_shapes(self):
        def check_not_none(value, name: str):
            if value is None:
                raise ValueError(f"Graph field '{name}' should not be None")

        check_not_none(self.n_node, 'n_node')
        check_not_none(self.n_edge, 'n_edge')

        if self.nodes is not None and self.nodes.shape[0] != self.n_node:
            raise ValueError(
                f"Nodes feature tensor should have the first dimension of size `n_node` ({self.nodes.shape[0]} instead of {self.n_node})"
            )

        if self.edges is not None:
            if self.senders.dim() != 1:
                raise ValueError("Graph `senders` should be 1D")

            if self.receivers.dim() != 1:
                raise ValueError("Graph `receivers` should be 1D")

            if self.edges.shape[0] != self.n_edge or \
                self.senders.shape[0] != self.n_edge or \
                self.receivers.shape[0] != self.n_edge:
                raise ValueError(f"Incorrect Graph `edges` shape")
