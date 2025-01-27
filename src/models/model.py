import tqdm
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

def train(model, data, epochs=100):
    model.train()
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in tqdm.tqdm(range(epochs)):
        if False: # type(data) is not Data:
            pass
            for batch in data:
                out, layer_weights = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        else:
            out = model(data=data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(model, data):
    model.eval()
    # if type(data) is not Data:
    #     correct = 0
    #     for batch in data:
    #         out, _ = model(batch)  
    #         pred = out.argmax(dim=1)
    #         correct += int((pred == batch.y).sum())
    #     return correct / len(data.dataset)
    pred = model(data=data).argmax(dim=1)
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / len(data.test_mask)
    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = int(train_correct) / len(data.train_mask)

    if isinstance(train_acc, Tensor):
        train_acc = train_acc.item()
    if isinstance(test_acc, Tensor):
        test_acc = test_acc.item()

    return train_acc, test_acc