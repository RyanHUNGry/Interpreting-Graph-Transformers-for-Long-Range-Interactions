from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain import Explanation
from torch import stack

class AttentionExplainer(ExplainerAlgorithm):
    def __init__(self, **kwargs):
        super().__init__()
        self.attention_weights = kwargs['attention_weights']

    def forward(self, model, x, edge_index, target, index=None, **kwargs) -> Explanation:
        """
        Computes the explanation of the trained model.
        """
        matrix = self.__average_attention_matrices()
        print(matrix)
        filtered_matrix = self.__filter_edges_from_attention_matrix(matrix, edge_index)

        return Explanation(edge_mask=self.attention_weights)
    
        # shortest path idea
        # mask out all the attentions that aren't actually an edge
        # takes in model attention weights, averaged so just one attention matrix
        # choose the 26th row, which attend sto all other nodes. Some edges don't exist, so filter them out. With
        # the remaining, choose the topK or select all, we'll see how many are selected and see if we want a constriant.
        # aggregates based on some condition, then generates edge mask, which should be same size as edge_index

    def __average_attention_matrices(self):
        stacked_matrices = stack(
            [self.attention_weights[layer][0] for layer in self.attention_weights] # There are L NxN matrices, stacked so LNxN
        )
        
        return stacked_matrices.mean(dim=0)
    
    def __filter_edges_from_attention_matrix(self, matrix, edge_index):
        pass

    def supports(self) -> bool:
        """Checks if the explainer supports the user-defined settings provided
        in :obj:`self.explainer_config`, :obj:`self.model_config`.
        """

        return True
    

# def max_reduce(matrix):

#     #apply max reduction along columns (dim = 1)
#     max_values, _ = matrix.max(dim=1)
#     return max_values

# #finds the weighted_average for one singular attention matrix

# #when summing across dim=0, output shows how much attention each node receives, when dim=1, it shows how much attention each node is giving

# def weighted_average_received(attention_matrix):
    
#     # Apply softmax across rows (dim=1) to normalize each row of the attention matrix
#     softmax_attention = F.softmax(attention_matrix, dim=1)
    
#     # Compute the weighted average across each row (dim=1) by summing
#     weighted_avg = softmax_attention.sum(dim=0)  # Sum along the columns
    
#     return weighted_avg

# def weighted_average_given(attention_matrix):

#     """Computes the weighted average of an attention matrix to show how much attention each node is giving to others"""
    
#     # Step 1: Apply softmax across rows (dim=1) to normalize attention
#     softmax_attention = F.softmax(attention_matrix, dim=1)
    
#     # Step 2: Compute the weighted average across rows (dim=1)
#     # Multiply each value by its respective column index (weighted sum)
#     weighted_avg = torch.matmul(softmax_attention, torch.arange(attention_matrix.size(1), dtype=torch.float32))

#     return weighted_avg

# def weighted_average_all_layers(function, matrices):

#     #store all the weighted averages per matrix (from each layer)
#     weighted_averages = []

#     #use the weighted_average function (single matrix use case) in a loop to collect all the weighted averages,
#     #and append to list

#     for matrix in matrices:
#         current_weighted_avg = function(matrix)
#         weighted_averages.append(current_weighted_avg)

#     # Compute the average of the weighted averages across all layers
#     avg_all_matrices= torch.stack(weighted_averages).mean(dim=0)

#     return weighted_averages, avg_all_matrices

# def top_k_nodes(matrix, top_k=0.1):

#     # Rank nodes by importance (highest first)
#     sorted_indices = torch.argsort(matrix, descending=True)

#     # Select the top K nodes (either as a percentage or fixed number)
#     if isinstance(top_k, float):
#         top_k = int(len(sorted_indices) * top_k)  # Percentage to number of nodes
#     top_nodes = sorted_indices[:top_k]
#     return top_nodes
