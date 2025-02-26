from src.models.model import train, test
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import DummyExplainer
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain.metric import groundtruth_metrics, fidelity, characterization_score
from tqdm import tqdm
import random

from torch_geometric.explain.algorithm import CaptumExplainer

from captum.attr import IntegratedGradients

class ExplainerPipeline:
    def __init__(self, data, num_classes, model, explainer: type, model_params = {}, explainer_params = {}, epochs=300, Hook=None):
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
            exp = explainer(**explainer_params, attention_weights = hook.attention_weights, data = self.data)
        elif explainer == CaptumExplainer:
            exp = explainer(attribution_method=IntegratedGradients)
        else:
            exp = explainer(**explainer_params) if not explainer == DummyExplainer else explainer()

        # explainer_config and model is passed to both Explainer and to ExplainerAlgorithm via ExplainerAlgorithm.connect(), 
        # and are accessible under ExplainerAlgorithm.model and ExplainerAlgorithm.explainer_config
        # module_config has attributes module_config.mode, module_config.task_level, module_config.return_type
        # these dictate how a certain ExplainerAlgorithm should operate, and unsupported configurations should be specified in ExplainerAlgorithm.supports()
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

    def get_explanation_metrics(self, node_idx: int, num_hops: int = 1):
        if node_idx not in self.explanations:
            raise ValueError("Node index has not been explained yet")
        
        _, _, _, ground_truth_mask = k_hop_subgraph(node_idx, num_hops=num_hops, edge_index=self.data.edge_index)
        return groundtruth_metrics(ground_truth_mask, self.explanations[node_idx].edge_mask, ["accuracy", "recall", "precision", "f1_score"], threshold=0.20)
    
    def get_explanation_fidelity(self, node_idx: int):
        if node_idx not in self.explanations:
            raise ValueError("Node index has not been explained yet")
        
        return fidelity(self.explainer, self.explanations[node_idx])
    
    def get_explanation_characterization_score(self, node_idx: int):
        if node_idx not in self.explanations:
            raise ValueError("Node index has not been explained yet")
        
        pos, neg = fidelity(self.explainer, self.explanations[node_idx])
        if pos == 0 or neg == 1:
            return "N/A"
        
        return characterization_score(pos, neg)

    def get_entire_explanation_fidelity(self, w_plus=0.5, w_minus=0.5, m=50, **kwargs):
        explanations = {}
        random_sample = random.sample(range(self.data.num_nodes), m)
        pos, neg = 0, 0
        for node_idx in tqdm(random_sample):
            explanations[node_idx] = self.explainer(x=self.data.x, edge_index=self.data.edge_index, index=node_idx, target=None, **kwargs)
            p, n = fidelity(self.explainer, explanations[node_idx])
            pos += p
            neg += n

        fid_plus, fid_neg = pos / len(explanations), neg / len(explanations)
        characterization_score = ((w_plus + w_minus) * fid_plus * (1 - fid_neg))/(w_minus * fid_plus + w_plus * (1 - fid_neg))

        return (fid_plus, fid_neg, characterization_score)
