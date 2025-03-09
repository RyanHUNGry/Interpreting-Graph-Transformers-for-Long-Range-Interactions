from tqdm import tqdm
from torch import Tensor
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Sigmoid, BCELoss
from torch.optim import Adam

def train(model, data, epochs=100):
    if not model.return_logits:
        raise ValueError("Model must return logits for training due to loss functions BCEWithLogitsLoss and CrossEntropyLoss expecting logits")
    
    if model.is_binary_classification:
        if getattr(model, "integrated_gradients", False):
            criterion = BCELoss()  # IG for binary classification requires probabilities
        else:
            criterion = BCEWithLogitsLoss()
    else:
        criterion = CrossEntropyLoss()


    model.train()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        out = model(data=data)

        # N x 1 tensor to N tensor to work with BCEWithLogitsLoss
        if isinstance(criterion, BCEWithLogitsLoss) or isinstance(criterion, BCELoss):
            train_out = out[data.train_mask].flatten()
        else:
            train_out = out[data.train_mask]

        train_labels = data.y[data.train_mask].float() if (isinstance(criterion, BCEWithLogitsLoss) or isinstance(criterion, BCELoss)) else data.y[data.train_mask]

        loss = criterion(train_out, train_labels)
        loss.backward()
        optimizer.step()

def test(model, data):
    model.eval()

    if model.is_binary_classification:
        logits = model(data=data)
        if getattr(model, "integrated_gradients", False):
            pred = (logits >= 0.5).int().flatten()  # IG case, assumes probabilities as input
        else:
            pred = (Sigmoid()(logits) >= 0.5).int().flatten()  # Convert logits to probabilities
    else:
        pred = model(data=data).argmax(dim=1)  # Multiclass classification


    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / len(data.test_mask)

    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = int(train_correct) / len(data.train_mask)

    if isinstance(train_acc, Tensor):
        train_acc = train_acc.item()
    if isinstance(test_acc, Tensor):
        test_acc = test_acc.item()

    return train_acc, test_acc
