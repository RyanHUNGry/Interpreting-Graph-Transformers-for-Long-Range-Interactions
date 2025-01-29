from src.models.model import train, test
from torch_geometric.explain import Explainer
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain.metric import groundtruth_metrics, fidelity

class ExplainerPipeline:
    def __init__(self, data, num_classes, model, explainer, model_params = {}, explainer_params = {}, epochs=300, Hook=None):
        self.data = data
        self.model = model(data, num_classes, **model_params)

        hook = None
        if Hook:
            hook = Hook(self.model)
        
        # train model
        train(self.model, data, epochs=epochs)

        # remove hooks so attention weights are not stored during test set evaluation
        if hook:
            hook.remove_hooks()

        if hook:
            exp = explainer(**explainer_params, attention_weights = hook.attention_weights)
        else:
            exp = explainer(**explainer_params)

        # initialize explainer with trained model
        self.explainer = Explainer(
            model = self.model,
            algorithm = exp,
            **explainer_params
        )

        # store generated individual explanations
        self.explanations = {}

    def get_accuracies(self):
        train_acc, test_acc = test(self.model, self.data)
        print(f"Train accuracy: {train_acc}")
        print(f"Test accuracy: {test_acc}")

    def explain(self, node_idx, **kwargs):
        self.explanations[node_idx] = self.explainer(x=self.data.x, edge_index=self.data.edge_index, index=node_idx, target=None, **kwargs)

    def get_explanation_accuracy(self, node_idx: int, num_hops: int = 1):
        if node_idx not in self.explanations:
            raise ValueError("Node index has not been explained yet")
        
        _, _, _, ground_truth_mask = k_hop_subgraph(node_idx, num_hops=num_hops, edge_index=self.data.edge_index)
        return groundtruth_metrics(ground_truth_mask, self.explanations[node_idx].edge_mask, "accuracy", threshold=0.20)
    
    def get_explanation_fidelity(self, node_idx: int):
        if node_idx not in self.explanations:
            raise ValueError("Node index has not been explained yet")
        
        return fidelity(self.explainer, self.explanations[node_idx])
