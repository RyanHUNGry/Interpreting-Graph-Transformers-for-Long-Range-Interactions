# Paying Attention

## Introduction
*Paying Attention* is an attention-based explainer framework for graph transformers, taking inspiration from attention-based explainers for traditional NLP transformers using global, multihead attention mechanisms.

#
we neeed to train GraphGPS on syn1
create a function to do a trained model




graph GPS ground truth, fidelity
Then, attention ground truth, fidelity ->
    While ATT does consider graph structure, it does not explain using node features and can only explain
    GAT models. Furthermore, in ATT it is not obvious which attention weights need to be used for edge importance, since a 1-hop neighbor of a node can also be a 2-hop neighbor of the same node due to
    cycles. Each edge’s importance is thus computed as the average attention weight across all layers.

Try an ensemble method where we extract the final layer's trained MPNN and use GraphGPS
Then combine with attention weights?