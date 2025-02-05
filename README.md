# Interpreting Graph Transformers for Long-Range Interactions

## Introduction
*Interpreting Graph Transformers for Long-Range Interactions* is an attention-based explainer framework for graph transformers, taking inspiration from attention-based explainers for traditional NLP transformers using global, multihead attention mechanisms.

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

        # shortest path idea
        # mask out all the attentions that aren't actually an edge
        # takes in model attention weights, averaged so just one attention matrix
        # choose the 26th row, which attend sto all other nodes. Some edges don't exist, so filter them out. With
        # the remaining, choose the topK or select all, we'll see how many are selected and see if we want a constriant.
        # aggregates based on some condition, then generates edge mask, which should be same size as edge_index

Try to do the BFS edge masking
Debug fidelity by generating a random explanation and see if it is 0 | 1
For the GNNExplainer on GPS, use some thresholding (top percentiel?) or some topK (percentage?) parameter to constrain the amount of data
Then, try to evaluate a good fidelity elbow 
