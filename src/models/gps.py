import torch
from torch.nn import ModuleList, LogSoftmax, Sigmoid
from torch_geometric.nn import GCNConv, GPSConv, global_add_pool
from src.models.attention.gps_conv_exposed_attention import GPSConvExposedAttention
import torch.nn as nn

"""
GPS model from the paper "Recipe for a General, Powerful, Scalable Graph Transformer"
by Ladislav Rampášek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini (https://arxiv.org/abs/2205.12454)

Tunable model hyperparameters include the number of hidden channels, positional encoding channels, attention heads, and hidden layers.
"""
class GPS(torch.nn.Module):
    def __init__(self, data, num_classes, hidden_channels=4, pe_channels=4, num_attention_heads=1, num_layers=1, observe_attention=False, return_logits=True, integrated_gradients=False):
        super().__init__()

        self.is_binary_classification = (num_classes == 2)
        self.return_logits = return_logits
        self.observe_attention = observe_attention
        self.integrated_gradients = integrated_gradients
        
        self.pe_channels = pe_channels
        self.pe_lin = nn.Linear(pe_channels, hidden_channels)
        self.pe_norm = nn.LayerNorm(hidden_channels)

        # for integrated gradients, PE is included in feature matrix so subtract PE dim
        if self.integrated_gradients:
            self.input_lin = nn.Linear(data.num_node_features - pe_channels, hidden_channels)
        else:
            self.input_lin = nn.Linear(data.num_node_features, hidden_channels)

        self.layers = ModuleList()
        hidden_channels *= 2
        for _ in range(num_layers):
            mpnn = GCNConv(hidden_channels, hidden_channels)

            if not self.observe_attention:
                transformer = GPSConv(hidden_channels, mpnn, heads=num_attention_heads)
            else:
                transformer = GPSConvExposedAttention(hidden_channels, mpnn, heads=num_attention_heads)

            self.layers.append(transformer)

        self.lin = nn.Linear(hidden_channels, num_classes if not self.is_binary_classification else 1)

        # loss functions cross entropy and binary cross entropy with logits expect logits
        # rather than probabilities, so we return logits by default
        if not return_logits and not self.is_binary_classification:
            self.probability_function = LogSoftmax(dim=-1)
        elif not return_logits and self.is_binary_classification:
            self.probability_function = Sigmoid()

    # Explainer.get_prediction() calls forward(x, edge_index, **kwargs) -> attached to the explanation object
    # User calls forward(data=data)
    def forward(self, *argv, data=None, **kwargs):
        # extract from function arguments, necessary for Explainer.get_prediction(), 
        # completely unrelated any explainer algorithm
        if argv:
            x = argv[0]
            edge_index = argv[1]
            pe = torch.empty(x.shape[0], 0)
            if not self.integrated_gradients:
                for p in kwargs:
                    # kwargs from parent call of Explainer are also used for forward of explainer algorithms
                    if isinstance(kwargs[p], torch.Tensor):
                        pe = torch.cat([pe, kwargs[p]], dim=1)
            # IntegratedGradients from Captum calls forward using argv but no **kwargs
            # since feature matrix is interpolated, PE must belong to matrix itself rather than
            # be passed as an additional attribute
            else:
                # idx 0 is original x, idx 1 is edge_index, idx 2 is positional encoding
                # x is interpolated x, which contains interpolated PE as well
                new_x = x[:, :-self.pe_channels]
                new_pe = x[:, -self.pe_channels:]

                x, pe = new_x, new_pe
        # extract from data object (user call)
        elif data:
            if not self.integrated_gradients:
                x = data.x
                edge_index = data.edge_index

                pe = torch.empty(x.shape[0], 0)
                if hasattr(data, "random_walk_pe"):
                    pe = torch.cat([pe, data.random_walk_pe], dim=1)
                if hasattr(data, "laplacian_eigenvector_pe"):
                    pe = torch.cat([pe, data.laplacian_eigenvector_pe], dim=1)
            else:
                x = data.x
                edge_index = data.edge_index

                new_x = x[:, :-self.pe_channels]
                new_pe = x[:, -self.pe_channels:]

                x, pe = new_x, new_pe

        if x.dtype is torch.float64:
            x = x.float()
        if pe.dtype is torch.float64:
            pe = pe.float()
            
        pe = self.pe_lin(pe)
        pe = self.pe_norm(pe)
        x = self.input_lin(x)
        x = torch.cat((x, pe), dim=1)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.lin(x)
        
        if not self.return_logits:
            return self.probability_function(x)
        else:
            return x
