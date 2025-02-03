from torch_geometric.datasets import Planetoid, TUDataset, LRGBDataset, ExplainerDataset, graph_generator
from torch_geometric.loader import DataLoader
from torch import cat, ones
import os
from torch_geometric.transforms import Compose, AddLaplacianEigenvectorPE, AddRandomWalkPE
from src.models.utils.utils import add_train_val_test_masks, add_arbitrary_feature, create_consecutive_mapping
from torch_geometric.utils import to_networkx

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'raw')

def load_clean_cora():
    cora_dataset = Planetoid(root=os.path.join(root_path, 'Planetoid'), name='Cora', pre_transform=Compose([AddLaplacianEigenvectorPE(5)]))
    return cora_dataset[0], cora_dataset.num_classes

def load_clean_bashapes(num_motifs = 4, laplacian_eigenvector_dimensions = 5, **kwargs):
    ba_shapes_dataset = ExplainerDataset(graph_generator=graph_generator.BAGraph(**kwargs), motif_generator='house', num_motifs=num_motifs, transform=Compose([AddLaplacianEigenvectorPE(laplacian_eigenvector_dimensions), add_train_val_test_masks, add_arbitrary_feature]))
    d = ba_shapes_dataset[0]
    return d, ba_shapes_dataset.num_classes, to_networkx(d, to_undirected=True)

def load_clean_pascalvoc_sp(idx=0):
    pascalvoc_sp_dataset = LRGBDataset(root=os.path.join(root_path, 'LRGB'), name='PascalVOC-SP', pre_transform=Compose([AddRandomWalkPE(5)]))
    d = pascalvoc_sp_dataset[idx]
    d = create_consecutive_mapping(d)
    d = add_train_val_test_masks(d)
    num_classes = len(d.y.unique())
    return d, num_classes, to_networkx(pascalvoc_sp_dataset[0], to_undirected=True)
