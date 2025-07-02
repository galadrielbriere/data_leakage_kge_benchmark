# Benchmarking Data Leakage on Link Prediction in Biomedical Knowledge Graph Embeddings

### Authors and paper
- **Galadriel Brière** (Aix Marseille Univ, INSERM, MMG, Marseille, France) 
- **Thomas Stosskopf** (TAGC, TGML, INSERM, UMR1090, Aix-Marseille University) 
- **Benjamin Loire** (Aix Marseille Univ, INSERM, MMG, Marseille, France) 
- **Anaïs Baudot** (Aix Marseille Univ, INSERM, MMG, Marseille, France; Barcelona Supercomputing Center, Barcelona, Spain)

📄 Brière, G., Stosskopf, T., Loire, B., & Baudot, A. (2025). *Benchmarking Data Leakage on Link Prediction in Biomedical Knowledge Graph Embeddings*. [bioRxiv](https://doi.org/10.1101/2025.01.23.634511)


## 🚧 **Note**

We are currently improving this implementation to create a standalone library and extend our framework with additional embedding models (including GNNs from PyTorch Geometric). Stay tuned for updates on [Knowledge Graph Autoencoder Training Environment (KGATE)](https://github.com/BAUDOTlab/KGATE/tree/main). 

## Introduction

This repository implements a systematic approach to assess and reduce data leakage in knowledge graph embedding-based link prediction over biomedical knowledge graphs. It offers a configurable pipeline for preprocessing, training, and evaluating popular knowledge graph embedding (KGE) models, with a particular focus on data leakage-aware benchmarking. The project relies on models implemented in TorchKGE, and training is handled via PyTorch and PyTorch-Ignite.

## Features

- Knowledge graph preprocessing with data leakage control.
  - Automatic detection and removal of semantical redundancy within training/test splits
  - Cold-start splits (head-based or tail-based)
- Customizable training and evaluation pipelines using configuration files.
- Supports KGE models from TorchKGE.

## Installation

### Prerequisites

- Python >= 3.10
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/galadrielbriere/drug_repurposing_kge_benchmark.git
   cd drug_repurposing_kge_benchmark
   ```

2. Create the environment and install dependencies:
   ```bash
   conda create --name torch_pyg python=3.10  
   conda activate torch_pyg
   pip install torch torchvision torchaudio
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   pip install torchkge
   pip install pandas matplotlib numpy pyyaml tqdm ignite pytorch-ignite
   ```


## Usage

### Configuration

Modify the `config.yaml` file to set parameters for KG preprocessing, training, and evaluation.

Bellow is an example of expected configuration file.

```yaml
common:  # Global parameters
  seed: 42  # Seed for reproducibility
  input_csv: "/path/to/knowledge_graph.tsv"  # Path to the input knowledge graph (TSV format with columns: "from", "to", "rel")
  out: "/path/to/results"  # Output directory for results
  run_kg_prep: true  # Run KG preprocessing
  run_training: false  # Train the model
  run_evaluation: false  # Evaluate the model on the test set
  run_inference: "/path/to/inference_kg.tsv"  # Optional: path to the inference KG (TSV with columns: "from", "to", "rel"). Set to false to disable.

clean_kg:  # Preprocessing settings
  remove_duplicates_triplets: true  # Recommended. Remove duplicate triples (e.g., a--ppi--b and b--ppi--a are considered duplicate triples)

  make_directed: true  # Optional. For specified relations, add reversed triplets (e.g., if a--r--b exists, also add b--r--a)
  make_directed_params:  # List of relation names to make directed
    - "disease_disease"
    - "drug_drug"
    - "protein_protein"

  check_DL1: true  # Recommended. Detect if there are redundant or Cartesian product relations in the KG.
  check_DL1_params: # See paper for details on DL1 detection criteria
    theta: 0.8   # Threshold for cartesian product relations
    theta1: 0.8  # Threshold for near-duplicate relations
    theta2: 0.8  # Threshold for near-reverse relations

  clean_train_set: true  # Recommended. Whether to remove redundant triplets involved in DL1 (redundant or Cartesian-product relations) from the training set

  permute_kg: false  # Optional. Whether to permute a specific relation (DL2 experiment, see paper)
  permute_kg_params:  # Relations to permute
    - "indication"  

  cold_start_split: 'head'  # Optional. Set to 'head' or 'tail' to perform a cold-start split on that position, or to false for a standard random split
  cold_start_split_param: "indication"  # Target relation for which the cold-start constraint applies



model:  # Model settings
  name: "TransE"  # Model name (any from TorchKGE)
  emb_dim: 200  # Embedding dimension
  margin: 1  # Margin for translational models 

sampler:  # Sampling strategy
  name: "Mixed"  # Sampler to use (Uniform, Positional, Bernouilli or Mixed)
  n_neg: 5 # Number of negatives to generate for each sampler and each fact (ignored for Positional)

optimizer:  # Optimizer configuration
  name: "Adam"  # Optimizer name (any from PyTorch)
  params:
    lr: 0.001 # Learning Rate
    weight_decay: 0.001  # Weight decay (L2 regularization)

lr_scheduler:  # Optional. Learning rate scheduler
  type: "CosineAnnealingWarmRestarts"  # Scheduler type (any from Pytorch)
  params:
    T_0: 10  # First cycle length
    T_mult: 2  # Multiplicative factor for cycle length

training:  # Training parameters
  max_epochs: 100  # Maximum number of epochs
  patience: 10  # Early stopping patience on validation MRR
  batch_size: 2048  # Training batch size
  eval_interval: 5  # Evaluation interval during training
  eval_batch_size: 32  # Evaluation batch size


evaluation:  # Evaluation settings – always computes filtered MRR and Hit@10
  made_directed_relations:  # Optional. Redundant/inverse relations (e.g. r and r_inv) to reproduce DL1 benchmark
    - "drug_drug"
    - "disease_disease"
    - "protein_protein"
    - "drug_drug_inv"
    - "disease_disease_inv"
    - "protein_protein_inv"
    # Can be left empty.
  target_relations:  # Relations to evaluate specifically
    - "indication"
    - "indication"
  thresholds:         # Optional. Same length as target_relations. For each relation, test facts are split by frequency in training:
                      # entities seen ≥ threshold vs < threshold with that relation
    - 10
    - 5

```


### Pipeline execution

```bash
python dev/run_training.py --config config.yaml 
```

## Reproducing Paper Results

### Knowledge Graph Availability

The Knowledge Graph used in our study is available on [Zenodo](https://zenodo.org/records/14577331?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjUwNTIxYWExLWExN2QtNDIwZi05YWUyLTI2MDI0OTM5MDlmOSIsImRhdGEiOnt9LCJyYW5kb20iOiI2YTY3NjY5ZTYwNWYwYWY3ZTZhMGJjOGY1YTJjNGIzYSJ9.ZSG9I-RjKuiUqiwpY66TvPLE6wAlvj3TdhjMvUCX-jUALSvFxa7tSBZpKFAgSJxLodX9ZhKQ_hJam4zd01cKyg). This Knowledge Graph is derived directly from the [Shepherd Knowledge Graph](https://zitniklab.hms.harvard.edu/projects/SHEPHERD/), with updated node identifiers to include node type.

### Configuration and Results 

All configuration files used to generate the results presented in the associated paper are included in this repository. The obtained results are organized into the following directories:

- **`DL1experiment`:** Results addressing data leakage caused by data redundancy during dataset splitting. This directory includes a subdirectory for runs with (`withDL1`) and without (`withoutDL1`) DL1.
- **`DL2experiment`:** Results exploring the use of node degree as illegitimate feature by KGE models (permutation experiment).
- **`DL3experiment`:** Results exploring cold-start split strategies.

Results on our inference dataset (drug repurposing for rare diseases) were obtained using models trained in the DL1 (`DL1experiment/withoutDL1`) setting or cold-start (`DL3experiment`) settings, and evaluated on a proprietary dataset provided by Orphanet, which we are unable to share publicly.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
