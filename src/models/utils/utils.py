from torch import arange, tensor, ones, unique, long
from sklearn.model_selection import train_test_split
import networkx as nx

def add_train_val_test_masks(data, train_size=0.8):
    idx = arange(data.y.shape[0])
    train_idx, test_idx = train_test_split(idx, train_size=train_size, stratify=data.y)
    data.train_mask = train_idx
    data.test_mask = test_idx

    return data

def add_arbitrary_feature(data):
    data.x = ones(data.y.shape[0], 1)
    return data

def create_consecutive_mapping(data):
    unique_labels = unique(data.y)
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
    data.y = tensor([label_mapping[label.item()] for label in data.y], dtype=long)
    return data
