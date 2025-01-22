from torch import randperm, zeros

def generate_masks(num_nodes, train_ratio, val_ratio):
    # Generate a random permutation of node indices
    indices = randperm(num_nodes)

    # Compute the sizes of each split
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    test_size = num_nodes - train_size - val_size

    # Assign indices to each split
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create masks
    train_mask = zeros(num_nodes, dtype=bool)
    val_mask = zeros(num_nodes, dtype=bool)
    test_mask = zeros(num_nodes, dtype=bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask

def add_train_val_test_masks(graph, num_nodes):
    train_mask, val_mask, test_mask = generate_masks(num_nodes, .10, .20)
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    return graph
