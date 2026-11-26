# LGMPS

LGMPS is a PyTorch implementation of "Linear-Time Generative Local-Global Modeling of Multi-Scale Dependencies for Personalized Search". The repository provides the model, training
and beam-search evaluation code, data-preparation adapters for KuaiSAR,
JDsearch, and Qilin, and reusable experiment configurations.

The main parallel-channel model combines session-local attention with a
Gated DeltaNet history path. Dense, category-aware feed-forward, MoE, PcMoE,
DeltaNet, Gated DeltaNet, Transformer, and aligned mixed-layer variants are
included.

## Installation

Create the provided Conda environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate lgmps
python -m pip install -e ".[data,dev,visualization]"
```

CUDA is optional for data preparation and CPU tests. GPU training and fused
kernels require a compatible CUDA environment. Configurations whose
`history_backend` is `fla` additionally require a compatible installation of
Flash Linear Attention.

## Data preparation

Datasets and generated artifacts are intentionally not distributed in this
repository. The adapters in [`data_proc/`](data_proc/README.md) convert
KuaiSAR, JDsearch, and Qilin into the shared
`session-sequence.v1` JSONL schema.

Run the built-in self-check or inspect the preparation commands from the
repository root:

```bash
python -m data_proc self-check
python -m data_proc --help
```

By default, dataset selection expects processed records and indexes under a
local `data/` directory. The expected paths are defined in
[`lgmps/configuration.py`](lgmps/configuration.py).

## Training and evaluation

Files in [`configs/`](configs/) describe shared model hyperparameters. Bind a
configuration to a public dataset at runtime:

```bash
lgmps-train --dataset kuaisar --config configs/lgmps_dense.json
```

Evaluate a checkpoint with the same dataset binding:

```bash
lgmps-evaluate \
  --dataset kuaisar \
  --config configs/lgmps_dense.json \
  --checkpoint /path/to/best.pt \
  --result-file /path/to/predictions.jsonl
```

Generated checkpoints, predictions, metrics, logs, and datasets remain local
and are ignored by Git.

## Verification

```bash
ruff check lgmps data_proc tests
ruff format --check lgmps data_proc tests
pytest -q
```

The test suite covers data contracts, chronological assembly, structured
attention masks, cached generation, checkpoint restoration, model variants,
retrieval metrics, and MoE routing.

## Repository layout

```text
lgmps/       model, training, inference, and visualization package
data_proc/   raw-data adapters and semantic-ID preparation
configs/     reusable model configurations
patches/     optional third-party integration patch
tests/       unit and integration tests
```
