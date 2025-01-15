import src.data.loader as loader
import src.models.gps as gps
import torch_geometric.data
import bertviz
# basic test on cora for dev purposes
cora_data, num_classes = loader.load_clean_cora()
print(isinstance(cora_data, torch_geometric.data.Data))
print(cora_data)
model = gps.GPS(cora_data, num_classes, pe_channels=5, hidden_channels=2, observe_attention=True)
w = gps.train(model, cora_data)

single_matrix = w[0][0]
print(w[0][0].shape)
bertviz.model_view
print(single_matrix.argmax(dim=1))
train_acc, test_acc = gps.test(model, cora_data)
print(f'Train Accuracy for Cora node-level classification: {train_acc}')
print(f'Test Accuracy for Cora node-level classification: {test_acc}')

