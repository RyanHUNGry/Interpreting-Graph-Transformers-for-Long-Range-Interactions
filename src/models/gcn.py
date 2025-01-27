import torch
from torch_geometric.data.data import Data
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import LogSoftmax, ReLU, Dropout

"""
GCN model from the paper "Semi-Supervised Classification with Graph Convolutional Networks" 
by Thomas N. Kipf and Max Welling (https://arxiv.org/pdf/1609.02907)

The GCN class and train/test methods, by default, performs node-level classification, but it will use the data instance
type to determine whether to perform graph-level classification if data is not an instance of torch_geometric.data.data.Data.
Graph-level classification will apply a summation readout on final node embeddings and then apply a linear layer to the pooled embeddings.

Tunable model hyperparameters include the number of hidden channels.
"""
class GCN(torch.nn.Module):
    def __init__(self, data, num_classes, hidden_channels=16, num_layers=2):
        super().__init__()
            
        self.conv1 = GCNConv(data.num_node_features, hidden_channels)
        self.relu = ReLU()
        self.dropout = Dropout()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            conv = GCNConv(hidden_channels, hidden_channels if i != num_layers - 2 else num_classes)
            relu = ReLU()
            dropout = Dropout()
            self.convs.append(conv)
            self.convs.append(relu) 
            self.convs.append(dropout)

        self.log_softmax = LogSoftmax(dim=-1)

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

        return self.log_softmax(x)
    