from torch_geometric.utils.convert import from_networkx
import src.models.gps as gps
from torch import tensor, float32
from torch_geometric.transforms import Compose, AddLaplacianEigenvectorPE
from src.models.utils.utils import add_train_val_test_masks

"""
Trains GraphGPS with exposed attention on GNNExplainer synthetic graphs.
"""
class SyntheticTrainer:
    def __init__(self, synthetic_generator, **kwargs):
        graph_networkx, role_ids, _ = synthetic_generator(**kwargs)
        self.graph_networkx = graph_networkx
        self.graph_pyg = from_networkx(self.graph_networkx)
        self.graph_pyg.y = tensor(role_ids)
        add_train_val_test_masks(self.graph_pyg, len(self.graph_pyg.y))
        self.graph_pyg.x = tensor([1] * len(self.graph_pyg.y)).to(float32).reshape(-1, 1)
        self.model = None

    def assign_encodings(self):
        transforms = Compose([AddLaplacianEigenvectorPE(5)])
        self.graph_pyg = transforms(self.graph_pyg)

    def initialize_model(self, **kwargs):
        num_classes = len(self.graph_pyg.y.unique())
        self.model = gps.GPS(self.graph_pyg, num_classes, **kwargs)
        return self.model
    
    def get_graph(self):
        return self.graph_pyg
