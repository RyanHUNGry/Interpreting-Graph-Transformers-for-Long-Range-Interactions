# Interpreting Graph Transformers for Long-Range Interactions

## Aim
*Interpreting Graph Transformers for Long-Range Interactions* is an attention-based explainer framework for graph transformers, taking inspiration from attention-based explainers for traditional NLP transformers using global, multihead attention mechanisms. 

We propose `AttentionExplainer`, an explainability algorithm leveraging the attention matrices during self-attention. We also benchmark our method against other explainability algorithms.

## Run
The framework is PyTorch based and leverages PyTorch Geometric for GNN operations.

We supply a Docker image to run our framework and benchmarks. The output of `run.py` will be serialized to disk at `outputs/results.json`. A running execution log will also be available via `stdout` to monitor progress.

> **⚠️ Notice:** Please expect a longer initial image pull. We find that baking the raw data into the image is faster than loading and processing at runtime. Training times can also vary given your machine's allocated CPU and RAM. GraphGPS, the transformer model, may take the longest due to its multi-head, global attention mechanism. Train and test accuracies can slightly differ as well, since our implementation randomly assigns train/test/validation masks. Explainer accuracy and fidelity computation will also vary given model sophistication and explainability method.

```
docker pull ghcr.io/ryanhungry/intepreting-graph-transformers-for-long-range-interactions:latest
docker run --name attention-explainer ghcr.io/ryanhungry/intepreting-graph-transformers-for-long-range-interactions:latest
```

To see `outputs/results.json` once the container finishes running the model, copy the file from the exited container to your local filesystem.

```
docker cp <container_id>:/usr/local/intepreting-graph-transformers-for-long-range-interactions/outputs/results.json <local_path>
```

## Configuration
We supply a `params.json` configuration file to pass in model parameters, organized by model and dataset type. The base hyperparameter configuration follows our setup in our paper, so the results can be reproduced without additional configuration.

To supply your own custom set of hyperparameters, clone this repository and edit `params.json` locally. Ensure field names are not changed, and that only JSON integer field types are used. Then, mount your `params.json` as a volume into the container during runtime.

```
docker run -v <local_path>/params.json:/params.json intepreting-graph-transformers-for-long-range-interactions:latest
```
