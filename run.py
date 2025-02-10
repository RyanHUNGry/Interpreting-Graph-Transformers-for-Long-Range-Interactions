from src.data import loader
from src.models.explainer.explainer_pipeline import ExplainerPipeline
from src.models.gcn import GCN
from src.models.gps import GPS
from src.models.utils.hooks import GPSHook
from src.models.explainer.attention_explainer import AttentionExplainer
from src.models.model import test
from src.models.explainer.gnn_explainer import GNNExplainer

from torch_geometric.explain.config import ModelConfig, ThresholdConfig
from torch_geometric.explain.algorithm import DummyExplainer

import json
import os

def main():
    parameters_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'params.json')

    with open(parameters_path, 'r') as file:
        parameters = json.load(file)

    res = {}
    # res["attention_explainer"] = run_attention_explainer(parameters['attention_explainer'])
    # res["dummy_explainer"] = run_dummy_explainer(parameters['dummy_explainer'])
    res["gnn_explainer"] = run_gnn_explainer(parameters['gnn_explainer'])

    res_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'outputs', 'results.json')

    with open(res_path, 'w') as f:
        json.dump(res, f, indent=4)

def run_attention_explainer(params):
    """
    Runs AttentionExplainer with GPS on BAShapes and PascalVOC-SP
    """
    res = {}

    ba_shapes_params = params["BAShapes"]["loader"]
    ba_shapes, ba_shapes_num_classes, _ = loader.load_clean_bashapes(**ba_shapes_params)

    ba_shapes_explainer_params = {
        'explanation_type': 'model',
        'node_mask_type': 'attributes',
        'edge_mask_type': 'object',
        'model_config': ModelConfig(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    }

    ba_shapes_gps_params = params["BAShapes"]["gps"]
    ba_shapes_explainer_pipelines = {
        "gps": ExplainerPipeline(
            ba_shapes,
            ba_shapes_num_classes,
            GPS,
            AttentionExplainer,
            model_params={
                **ba_shapes_gps_params,
                'observe_attention': True
            },
            explainer_params=ba_shapes_explainer_params,
            epochs=400,
            Hook=GPSHook
        )
    }

    gps_train_acc, gps_test_acc = test(ba_shapes_explainer_pipelines["gps"].model, ba_shapes)

    res["BAShapes"] = {
        "gps": {
            "train_accuracy": gps_train_acc,
            "test_accuracy": gps_test_acc
        }
    }

    for model in ba_shapes_explainer_pipelines:
        for node in params["BAShapes"]["nodes_to_explain"]:
            ba_shapes_explainer_pipelines[model].explain(node, laplacian_eigenvector_pe=ba_shapes.laplacian_eigenvector_pe, attention_computation_method="shortest_path", top_k=5)

    for model in ba_shapes_explainer_pipelines:
        metrics = []
        for node in params["BAShapes"]["nodes_to_explain"]:
            metrics.append([{"node": node, "accuracy": ba_shapes_explainer_pipelines[model].get_explanation_accuracy(node)}])
        res["BAShapes"][model]["ground_truth_accuracy"] = metrics

    for model in ba_shapes_explainer_pipelines:
        pos, neg, characterization = ba_shapes_explainer_pipelines[model].get_entire_explanation_fidelity(laplacian_eigenvector_pe=ba_shapes.laplacian_eigenvector_pe, disable_tqdm=True)
        res["BAShapes"][model]["fidelity"] = {
            "positive": pos,
            "negative": neg,
            "characterization": characterization
        }

    pascalvoc_sp_params = params["PascalVOC-SP"]["loader"]
    pascalvoc_sp, pascalvoc_sp_num_classes, _ = loader.load_clean_pascalvoc_sp(**pascalvoc_sp_params)

    pascalvoc_sp_explainer_params = {
        'explanation_type': 'model',
        'node_mask_type': 'attributes',
        'edge_mask_type': 'object',
        'model_config': ModelConfig(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        )
    }

    pascalvoc_sp_params = params["PascalVOC-SP"]["gps"]
    pascalvoc_sp_explainer_pipelines = {
        "gps": ExplainerPipeline(
            pascalvoc_sp,
            pascalvoc_sp_num_classes,
            GPS,
            AttentionExplainer,
            model_params={
                **pascalvoc_sp_params,
                'observe_attention': True,
            },
            explainer_params=pascalvoc_sp_explainer_params,
            epochs=500,
            Hook=GPSHook
        ),
    }

    gps_train_acc, gps_test_acc = test(pascalvoc_sp_explainer_pipelines["gps"].model, pascalvoc_sp)

    res["PascalVOC-SP"] = {
        "gps": {
            "train_accuracy": gps_train_acc,
            "test_accuracy": gps_test_acc
        }
    }

    for model in pascalvoc_sp_explainer_pipelines:
        pos, neg, characterization = pascalvoc_sp_explainer_pipelines[model].get_entire_explanation_fidelity(random_walk_pe=pascalvoc_sp.random_walk_pe, disable_tqdm=True)
        res["PascalVOC-SP"][model]["fidelity"] = {
            "positive": pos,
            "negative": neg,
            "characterization": characterization
        }

    return res

def run_dummy_explainer(params):
    """
    Runs DummyExplainer with GPS and GCN on BAShapes
    """
    res = {}

    ba_shapes_params = params["BAShapes"]["loader"]
    ba_shapes, ba_shapes_num_classes, _ = loader.load_clean_bashapes(**ba_shapes_params)

    ba_shapes_explainer_params = {
        'explanation_type': 'model',
        'node_mask_type': 'attributes',
        'edge_mask_type': 'object',
        'model_config': ModelConfig(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    }

    ba_shapes_gps_params = params["BAShapes"]["gps"]
    ba_shapes_gcn_params = params["BAShapes"]["gcn"]
    ba_shapes_explainer_pipelines = {
        "gps": ExplainerPipeline(
            ba_shapes,
            ba_shapes_num_classes,
            GPS,
            DummyExplainer,
            model_params={
                **ba_shapes_gps_params,
                'observe_attention': True
            },
            explainer_params={
                **ba_shapes_explainer_params,
                "threshold_config": ThresholdConfig(threshold_type='topk', value=10)
            },
            epochs=4000
        ),
        "gcn": ExplainerPipeline(
            ba_shapes,
            ba_shapes_num_classes,
            GCN,
            DummyExplainer,
            model_params=ba_shapes_gcn_params,
            explainer_params=ba_shapes_explainer_params,
            epochs=4000
        )
    }

    gps_train_acc, gps_test_acc = test(ba_shapes_explainer_pipelines["gps"].model, ba_shapes)
    gcn_train_acc, gcn_test_acc = test(ba_shapes_explainer_pipelines["gcn"].model, ba_shapes)

    res["BAShapes"] = {
        "gps": {
            "train_accuracy": gps_train_acc,
            "test_accuracy": gps_test_acc
        },
        "gcn": {
            "train_accuracy": gcn_train_acc,
            "test_accuracy": gcn_test_acc
        }
    }

    for model in ba_shapes_explainer_pipelines:
        for node in params["BAShapes"]["nodes_to_explain"]:
            ba_shapes_explainer_pipelines[model].explain(node, laplacian_eigenvector_pe=ba_shapes.laplacian_eigenvector_pe, attention_computation_method="shortest_path", top_k=5)

    for model in ba_shapes_explainer_pipelines:
        metrics = []
        for node in params["BAShapes"]["nodes_to_explain"]:
            metrics.append([{"node": node, "accuracy": ba_shapes_explainer_pipelines[model].get_explanation_accuracy(node)}])
        res["BAShapes"][model]["ground_truth_accuracy"] = metrics

    for model in ba_shapes_explainer_pipelines:
        pos, neg, characterization = ba_shapes_explainer_pipelines[model].get_entire_explanation_fidelity(laplacian_eigenvector_pe=ba_shapes.laplacian_eigenvector_pe, disable_tqdm=True)
        res["BAShapes"][model]["fidelity"] = {
            "positive": pos,
            "negative": neg,
            "characterization": characterization
        }

    return res

def run_gnn_explainer(params):
    """
    Runs GNNExplainer with GPS and GCN on BAShapes and PascalVOC-SP
    """

    res = {}

    ba_shapes_params = params["BAShapes"]["loader"]
    ba_shapes, ba_shapes_num_classes, _ = loader.load_clean_bashapes(**ba_shapes_params)

    ba_shapes_explainer_params = {
        'explanation_type': 'model',
        'node_mask_type': 'attributes',
        'edge_mask_type': 'object',
        'model_config': ModelConfig(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    }

    ba_shapes_gps_params = params["BAShapes"]["gps"]
    ba_shapes_gcn_params = params["BAShapes"]["gcn"]
    ba_shapes_explainer_pipelines = {
        "gps": ExplainerPipeline(
            ba_shapes,
            ba_shapes_num_classes,
            GPS,
            GNNExplainer,
            model_params={
                **ba_shapes_gps_params,
                'observe_attention': True
            },
            explainer_params=ba_shapes_explainer_params,
            epochs=400,
            Hook=GPSHook
        ),
        "gcn": ExplainerPipeline(
            ba_shapes,
            ba_shapes_num_classes,
            GCN,
            GNNExplainer,
            model_params={
                **ba_shapes_gcn_params
            },
            explainer_params={
                **ba_shapes_explainer_params,
                "threshold_config": ThresholdConfig(threshold_type='topk', value=10)
            },
            epochs=4000
        )
    }

    gps_train_acc, gps_test_acc = test(ba_shapes_explainer_pipelines["gps"].model, ba_shapes)
    gcn_train_acc, gcn_test_acc = test(ba_shapes_explainer_pipelines["gcn"].model, ba_shapes)

    res["BAShapes"] = {
        "gps": {
            "train_accuracy": gps_train_acc,
            "test_accuracy": gps_test_acc
        },
        "gcn": {
            "train_accuracy": gcn_train_acc,
            "test_accuracy": gcn_test_acc
        }
    }

    for model in ba_shapes_explainer_pipelines:
        for node in params["BAShapes"]["nodes_to_explain"]:
            ba_shapes_explainer_pipelines[model].explain(node, laplacian_eigenvector_pe=ba_shapes.laplacian_eigenvector_pe, attention_computation_method="shortest_path", top_k=5)

    for model in ba_shapes_explainer_pipelines:
        metrics = []
        for node in params["BAShapes"]["nodes_to_explain"]:
            metrics.append([{"node": node, "accuracy": ba_shapes_explainer_pipelines[model].get_explanation_accuracy(node)}])
        res["BAShapes"][model]["ground_truth_accuracy"] = metrics

    for model in ba_shapes_explainer_pipelines:
        pos, neg, characterization = ba_shapes_explainer_pipelines[model].get_entire_explanation_fidelity(laplacian_eigenvector_pe=ba_shapes.laplacian_eigenvector_pe, disable_tqdm=True)
        res["BAShapes"][model]["fidelity"] = {
            "positive": pos,
            "negative": neg,
            "characterization": characterization
        }

    pascalvoc_sp_params = params["PascalVOC-SP"]["loader"]
    pascalvoc_sp, pascalvoc_sp_num_classes, _ = loader.load_clean_pascalvoc_sp(**pascalvoc_sp_params)

    pascalvoc_sp_explainer_params = {
        'explanation_type': 'model',
        'node_mask_type': 'attributes',
        'edge_mask_type': 'object',
        'model_config': ModelConfig(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        )
    }

    pascalvoc_sp_gps_params = params["PascalVOC-SP"]["gps"]
    pascalvoc_sp_gcn_params = params["PascalVOC-SP"]["gcn"]
    pascalvoc_sp_explainer_pipelines = {
        "gps": ExplainerPipeline(
            pascalvoc_sp,
            pascalvoc_sp_num_classes,
            GPS,
            GNNExplainer,
            model_params={
                **pascalvoc_sp_gps_params,
                'observe_attention': True
            },
            explainer_params={
                **pascalvoc_sp_explainer_params,
                "threshold_config": ThresholdConfig(threshold_type='topk', value=10)
            },
            epochs=500
        ),
        "gcn": ExplainerPipeline(
            pascalvoc_sp,
            pascalvoc_sp_num_classes,
            GCN,
            GNNExplainer,
            model_params={
                **pascalvoc_sp_gcn_params
            },
            explainer_params={
                **pascalvoc_sp_explainer_params,
                "threshold_config": ThresholdConfig(threshold_type='topk', value=10)
            },
            epochs=2000
        )
    }

    gps_train_acc, gps_test_acc = test(pascalvoc_sp_explainer_pipelines["gps"].model, pascalvoc_sp)
    gcn_train_acc, gcn_test_acc = test(pascalvoc_sp_explainer_pipelines["gcn"].model, pascalvoc_sp)

    res["PascalVOC-SP"] = {
        "gps": {
            "train_accuracy": gps_train_acc,
            "test_accuracy": gps_test_acc
        },
        "gcn": {
            "train_accuracy": gcn_train_acc,
            "test_accuracy": gcn_test_acc
        }
    }

    for model in pascalvoc_sp_explainer_pipelines:
        pos, neg, characterization = pascalvoc_sp_explainer_pipelines[model].get_entire_explanation_fidelity(random_walk_pe=pascalvoc_sp.random_walk_pe, disable_tqdm=True)
        res["PascalVOC-SP"][model]["fidelity"] = {
            "positive": pos,
            "negative": neg,
            "characterization": characterization
        }

    return res

if __name__ == '__main__':
    main()
