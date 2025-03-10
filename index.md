# Interpreting Graph Transformers for Long Range Interactions

## Introduction

Graphs are used far and wide in every industry imaginable - they're used to model social networks, biological networks, supply chain systems, and more. 

Machine learning models such as graph neural networks 

## Methods

### Data

We use two datasets to benchmark our proposed explainers below, **BAShapes** and **PascalVOC-SP**.

#### BAShapes

BAShapes is a synthetically generated graph that has 25 central nodes, with 5 edges per node, with 10 house motifs attached. 

#### PascalVOC-SP

**PascalVOC-SP** is a node-classification graph dataset originating from semantic segmentation in computer vision. We use this dataset due to its presence of long-range interactions. This dataset has a multitude of graphs, so we selected one for node level, binary classification.

![PascalVOC-SP Graph Visualization](assets/PascalVOC-SP%Graph.PNG)

## Metrics

We use a multitude of metrics to measure and quantify the efficacy and robustness of our methods to see where various explainers excel.

### Ground Truth, Recall, Precision

We use **ground truth accuracy**, **recall**, and **precision** only for BAShapes, as generated explanations on this dataset can be compared against a known explanation truth mask.

Ground truth accuracy refers to how accurate our generated explanation subgraphs are to the true mask. High ground truth accuracy means that an explainer can capture the proper house motif nodes in a generated explanation.

In addition to ground truth accuracy, we use recall and precision to paint a complete picture of our explainers' performance. Recall measures the proportion of accurate nodes captured in an explanation to all the true nodes. 

Precision captures the proportion of correct house motif nodes to all the predicted nodes in an explanation. This means that explainers with high precision will capture the correct the proper house motif nodes while ignoring the causally irrelevant ones.

F1-Score is a metric that provides a balanced measure of both recall and precision when evaluating the quality of an explanation.

### Fidelity and Characterization Scores

We use a metric called **Fidelity** to quantify how "necessary" and "sufficient" our generated explanations are. **Fidelity** has two submetrics for this - **Fid+** and **Fid-**.

* **Fid+** quantifies a "necessary" explanation. An explanation is deemed necessary if the model's prediction changes when it is removed from the initial graph.
* **Fid-** quantifies a "sufficient" explanation. An explanation is deemed sufficient if the model comes to the same initial prediction with just the explanation subgraph as the entire graph.

Characterization scores combine both Fidelity measures into one metric, where a higher value indicates better performance.

## Results

Results Table (Placeholder for now)

| Metric 1 | Metric 2 | Metric 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |
| Data 7   | Data 8   | Data 9   |


Add Heat Maps and Subgraph Visualizations


## Discussion

## Conclusion

The importance of neural network explainability has grown alongside the increasing complexity of both data and model architectures. Specifically, graph neural network explainability aims to shed light on the decision-making processes of models that incorporate both features and relational structures during training. However, there is a notable gap in the literature regarding the explainability of graph transformers. To address this, we propose two methods: **AttentionExplainer** and **IGExplainer**. The former utilizes trained attention weights to greedily generate subgraph explanations, while the latter applies integrated gradients to compute edge attribution for each edge in the graph. Our approach is directly applicable to any graph transformer model that employs self-attention, offering a more appropriate explanation framework for this architecture compared to existing explainer techniques that are primarily designed for message-passing neural networks. Our explainer algorithms provide efficient and vital interpretations of graph transformer decision-making, providing critical model transparency, fairness, ethicality, and legal adherence.

## References
