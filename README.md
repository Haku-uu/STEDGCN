## README.md

````markdown
# STEDGCN: Spatiotemporal Trend-Event Decoupling Graph Convolutional Network for Traffic Flow Prediction

This repository provides the official implementation of **STEDGCN**, a Spatiotemporal Trend-Event Decoupling Graph Convolutional Network for traffic flow prediction.

STEDGCN is designed to model complex traffic dynamics by explicitly separating traffic sequences into low-frequency trend components and high-frequency event components. The model further constructs frequency-specific spatiotemporal representations through dual-path temporal modeling, adaptive graph construction, diffusive graph propagation, and gated spatiotemporal fusion.

## Overview

Traffic flow prediction is a fundamental task in Intelligent Transportation Systems (ITS). Existing spatiotemporal graph neural networks often model traffic sequences in a unified temporal space, which may cause interference between long-term periodic trends and short-term abrupt fluctuations. To address this issue, STEDGCN introduces a frequency-aware trend-event decoupling framework.

The main components of STEDGCN include:

- **Temporal Signal Separator (TSS)**  
  Decomposes raw traffic sequences into low-frequency trend signals and high-frequency event signals.

- **Dual-Frequency Spatiotemporal Encoder (DFSE)**  
  Encodes trend and event components through two frequency-specific branches.

- **Trend-Adaptive Graph Convolutional Network (TA-GCN)**  
  Captures stable and trend-driven spatial dependencies.

- **Event-Aware Graph Convolutional Network (EA-GCN)**  
  Models event-induced spatial perturbations and abrupt traffic changes.

- **Diffusive Graph Signal Propagation (D-GSP)**  
  Performs graph-based information propagation over adaptive spatial structures.

- **Fusion-Gated Spatiotemporal Decoder (FGSD)**  
  Integrates trend and event representations through gated feature selection and cross-path fusion.

## Repository Structure

```text
STEDGCN/
├── model.py                 # Main STEDGCN model
├── stedgcn_attention.py      # Temporal multi-head attention modules
├── train.py                 # Training script
├── test.py                  # Testing script
├── util.py                  # Data loading and evaluation utilities
├── ranger21.py              # Ranger optimizer
├── README.md
└── requirements.txt
````

## Requirements

The implementation is based on PyTorch. Please install the required package by:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for training.

## Datasets

This repository does not include datasets. Please download the public traffic flow datasets commonly used in traffic forecasting papers, such as:

* PeMS03
* PeMS04
* PeMS07
* PeMS08

These datasets are widely used in representative traffic forecasting studies such as STGCN, DCRNN, STSGCN, STFGNN, Graph WaveNet, and related spatiotemporal graph neural network papers.

Please preprocess the datasets into the following structure:

```text
data/
├── PEMS03/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── PEMS04/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── PEMS07/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
└── PEMS08/
    ├── train.npz
    ├── val.npz
    └── test.npz
```

Each `.npz` file should contain:

```text
x: historical input sequences
y: future prediction targets
```

The expected input format follows the standard traffic forecasting setting:

```text
[B, T, N, C]
```

where:

* `B` is the batch size
* `T` is the number of historical time steps
* `N` is the number of traffic sensors
* `C` is the number of input features

## Training

To train STEDGCN on PeMS03:

```bash
python train.py --data PEMS03 --device cuda:0
```

To train on other datasets:

```bash
python train.py --data PEMS04 --device cuda:0
python train.py --data PEMS07 --device cuda:0
python train.py --data PEMS08 --device cuda:0
```

Main arguments:

```text
--data          Dataset name
--device        Training device
--input_dim     Number of input features
--channels      Hidden feature dimension
--num_nodes     Number of traffic sensors
--input_len     Historical sequence length
--output_len    Forecasting horizon
--batch_size    Batch size
--learning_rate Learning rate
--dropout       Dropout rate
--epochs        Number of training epochs
```

## Testing

After training, specify the saved checkpoint and run:

```bash
python test.py --data PEMS03 --device cuda:0 --checkpoint ./logs/PEMS03/best_model.pth
```

Please replace the checkpoint path with the actual path generated during training.

## Evaluation Metrics

The model is evaluated using three common traffic forecasting metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)

These metrics are computed after inverse normalization.

## Citation

If this code is useful for your research, please cite our paper:

```bibtex
@article{stedgcn,
  title={A Spatiotemporal Trend-Event Decoupling Graph Convolutional Network for Traffic Flow Prediction},
  author={},
  journal={},
  year={}
}
```

