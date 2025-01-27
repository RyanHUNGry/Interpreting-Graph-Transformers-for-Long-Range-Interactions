from torch import arange, tensor, ones
from sklearn.model_selection import train_test_split

def add_train_val_test_masks(data, train_size=0.8):
    idx = arange(data.y.shape[0])
    train_idx, test_idx = train_test_split(idx, train_size=train_size, stratify=data.y)
    data.train_mask = train_idx
    data.test_mask = test_idx

    return data

def add_arbitrary_feature(data):
    data.x = ones(data.y.shape[0], 1)
    return data
