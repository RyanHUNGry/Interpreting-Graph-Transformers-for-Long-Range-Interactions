import torch
from torch_geometric.data.data import Data
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import LogSoftmax, ReLU, Dropout, Sigmoid

"""
GCN model from the paper "Semi-Supervised Classification with Graph Convolutional Networks" 
by Thomas N. Kipf and Max Welling (https://arxiv.org/pdf/1609.02907)

Tunable model hyperparameters include the number of hidden channels and number of layers.
"""
class GCN(torch.nn.Module):
    def __init__(self, data, num_classes, hidden_channels=16, num_layers=2, return_logits=True):
        super().__init__()
        
        self.is_binary_classification = (num_classes == 2)
        self.return_logits = return_logits

        self.conv1 = GCNConv(data.num_node_features, hidden_channels)
        self.relu = ReLU()
        self.dropout = Dropout()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            conv = GCNConv(hidden_channels, hidden_channels if i != num_layers - 2 else num_classes if not self.is_binary_classification else 1)
            relu = ReLU()
            dropout = Dropout()
            self.convs.append(conv)
            self.convs.append(relu) 
            self.convs.append(dropout)

        # loss functions cross entropy and binary cross entropy with logits expect logits
        # rather than probabilities, so we return logits by default
        if not return_logits and not self.is_binary_classification:
            self.probability_function = LogSoftmax(dim=-1)
        elif not return_logits and self.is_binary_classification:
            self.probability_function = Sigmoid()

    # Explainer.get_prediction() calls forward(x, edge_index, **kwargs)
    # Explainer.get_target() calls forward(prediction) where prediction = Explainer.get_prediction(...)
    # For non-explainability usage, specify data as a keyword argument
    def forward(self, *argv, data=None, **kwargs):
        if argv:
            x = argv[0]
            edge_index = argv[1]
        else:
            x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        for layer in self.convs:
            x = layer(x, edge_index) if isinstance(layer, GCNConv) else layer(x)

        if not self.return_logits:
            return self.probability_function(x)
        else:
            return x
    