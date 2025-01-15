from torch_geometric.datasets import Planetoid, TUDataset, LRGBDataset
from torch_geometric.loader import DataLoader
from torch import cat
import os
from torch_geometric.transforms import Compose, AddLaplacianEigenvectorPE, AddRandomWalkPE

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'raw')

def load_clean_cora(transformations=None):
    cora_dataset = Planetoid(root=os.path.join(root_path, 'Planetoid'), name='Cora', pre_transform=Compose([AddLaplacianEigenvectorPE(5)]))
    print(type(cora_dataset))
    return cora_dataset[0], cora_dataset.num_classes
