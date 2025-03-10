## Introduction

## Methods

### Data

#### BA-Shapes

#### PascalVOC-SP

#### AttentionExplainer

## Metrics

We use a multitude of metrics to measure and quantify the efficacy and robustness of our methods to see where various explainers excel.

### Ground Truth, Recall, Precision

We use ground truth accuracy, recall, and precision only for BAShapes, as generated explanations on this dataset can be compared against a known explanation truth mask.

Ground truth accuracy refers to how accurate our generated explanation subgraphs are to the true mask. High ground truth accuracy means that an explainer can capture the proper house motif nodes in a generated explanation.

In addition to ground truth accuracy, we use recall and precision to paint a complete picture of our explainers' performance. Recall measures the proportion of accurate nodes captured in an explanation to all the true nodes. 

Precision captures the proportion of correct house motif nodes to all the predicted nodes in an explanation. This means that explainers with high precision will capture the correct the proper house motif nodes while ignoring the causally irrelevant ones.

### Fidelity and Characterization Scores

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

## References
