from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain import Explanation
from torch_geometric.explain.config import ExplanationType, ModelTaskLevel, ModelReturnType
from torch_geometric.utils import to_networkx
from torch import stack, zeros, topk, full, tensor, maximum
import numpy as np

from typing import Literal

from networkx import shortest_path

class AttentionExplainer(ExplainerAlgorithm):
    def __init__(self, **kwargs):
        super().__init__()
        self.attention_weights = kwargs['attention_weights']
        self.data = to_networkx(kwargs['data']) # network x graph, not pyg

    # def __compute_neighbor_explanation_hops_and_top_k(self, edge_index, index, hops, top_k=None) -> Explanation:
    #     if hops == 1: # one-hop neighborhood
    #         return self.__compute_neighbor_explanation_top_k(edge_index, index, top_k)

    #     matrix = self.__average_attention_matrices()
    #     filtered_matrix = self.__filter_edges_from_attention_matrix(matrix, edge_index)

    def __compute_neighbor_explanation_top_k(self, edge_index, index, top_k=None) -> Explanation:
        matrix = self.__average_attention_matrices()
        filtered_matrix = self.__filter_edges_from_attention_matrix(matrix, edge_index)

        row = filtered_matrix[index].squeeze() # node index i attends to node j in this vector

        if top_k and top_k < len(row):
            top_k_tensor = zeros(len(row))
            top_values, top_ind = topk(row, top_k)
            top_k_tensor[top_ind] = top_values
            row = top_k_tensor

        edge_mask = zeros(edge_index.shape[1])

        # Vectorized approach
        i_vals, j_vals = edge_index[0], edge_index[1]  # Get node pairs
        mask_j = (j_vals == index)  # Mask where target node is index

        # Apply row values based on conditions
        edge_mask[mask_j] = row[i_vals[mask_j]]

        return Explanation(edge_mask=edge_mask)
    
    def __compute_neighbor_explanation_shortest_path(self, edge_index, index, top_k = 5):
        matrix = self.__average_attention_matrices()

        # Get the top k nodes that node index attends to based off weight
        _, c = topk(matrix[index], top_k)
        r = full((c.shape[1], ), index.item())
        c = c.flatten()
        coords = stack([r, c], dim=1)

        edge_mask = zeros(edge_index.shape[1]).int()

        # loops through each edge in shortest path
        for (r, c) in coords:
            r = r.item()
            c = c.item()

            shortest_path = tensor(self.__find_shortest_path(source=r, target=c))
            edges = stack([shortest_path[:-1], shortest_path[1:]]).flip(0)
            # edges = cat([edges, edges.flip(0)], dim=1) # uncomment to add bidirectional edges in explanation subgraph

            # generate edge mask from shortest_path
            curr_edge_mask = (edge_index.unsqueeze(2) == edges.unsqueeze(1)).all(dim=0).any(dim=1).int()
            edge_mask = maximum(edge_mask, curr_edge_mask)

        return Explanation(edge_mask=edge_mask)

    def forward(self, model, x, edge_index, target, index=None, **kwargs: dict[Literal["top_k", "shortest_path"] | None]) -> Explanation:
        """
        Computes the explanation of the trained model.
        """
        attention_computation_method = kwargs.get("attention_computation_method", "top_k") # default topk method
        top_k = kwargs.get("top_k", 5)
        if attention_computation_method == "top_k":
            return self.__compute_neighbor_explanation_top_k(edge_index, index, top_k)
        elif attention_computation_method == "shortest_path":
            return self.__compute_neighbor_explanation_shortest_path(edge_index, index, top_k)
        
    def __find_shortest_path(self, source, target):
        paths = shortest_path(self.data, source, target)
        return paths # generator

    def __average_attention_matrices(self):
        stacked_matrices = stack([
            (self.attention_weights[layer][0] - self.attention_weights[layer][0].min()) / 
            (self.attention_weights[layer][0].max() - self.attention_weights[layer][0].min())
            for layer in self.attention_weights
        ])  # There are L NxN matrices, so new stacked dimension is (LXN)xN
        
        return stacked_matrices.mean(dim=0)
    
    def __filter_edges_from_attention_matrix(self, matrix, edge_index):
        """
        Returns attentions weights that correspond to the edges in the graph. 
        """
        n = matrix.shape[0]
        new_matrix = zeros(n, n)

        new_matrix[edge_index[0], edge_index[1]] = matrix[edge_index[0], edge_index[1]]
        return new_matrix

    def supports(self) -> bool:
        """Checks if the explainer supports the user-defined settings provided
        in :obj:`self.explainer_config`, :obj:`self.model_config`.
        """

        explanation_type = self.explainer_config.explanation_type
        task_level = self.model_config.task_level
        return_type = self.model_config.return_type

        # Curr attention explainer only supports model explanation
        if explanation_type == ExplanationType.phenomenon:
            return False
        
        # only node (instance) level explanations are supported
        if task_level == ModelTaskLevel.edge or task_level == ModelTaskLevel.graph:
            return False
        
        # expect GCN and GPS to return logits
        if return_type != ModelReturnType.raw:
            return False

        return True
