from torch_geometric.datasets import Planetoid, TUDataset, LRGBDataset, ExplainerDataset, graph_generator
from torch_geometric.loader import DataLoader
from torch import cat, ones
import os
from torch_geometric.transforms import Compose, AddLaplacianEigenvectorPE, AddRandomWalkPE
from src.models.utils.utils import add_train_val_test_masks

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'raw')

def load_clean_cora():
    cora_dataset = Planetoid(root=os.path.join(root_path, 'Planetoid'), name='Cora', pre_transform=Compose([AddLaplacianEigenvectorPE(5)]))
    return cora_dataset[0], cora_dataset.num_classes

def load_clean_bashapes():
    ba_shapes_dataset = ExplainerDataset(graph_generator=graph_generator.BAGraph(num_nodes=300, num_edges=5), motif_generator='house', num_motifs=80, transform=Compose([AddLaplacianEigenvectorPE(5), add_train_val_test_masks]))
    d = ba_shapes_dataset[0]
    d.x = ones((len(d.y), 1))
    return d, ba_shapes_dataset.num_classes
