# GNNFingers: Verifying GNN Ownership via Fingerprinting

This project implements the paper **"GNNFingers: A Fingerprinting Framework for Verifying Ownerships of Graph Neural Networks"** (You et al., 2024). The goal is to detect unauthorized use of GNNs via graph fingerprinting without altering the original model.

## Structure
GNNFingers/
├── config/ # YAML configs
├── data/ # Dataset helpers / download
├── models/ # GCN, GraphSAGE, SimGNN, ...
├── fingerprint/ # FingerprintBuilder & utils
├── verifier/ # Univerifier MLP
├── training/ # Alternating-train engine
├── evaluation/ # Metrics & baselines
├── utils/ # Logging, seeds, misc
├── main.py # CLI entry-point
└── README.md

## Getting Started

```bash
pip install -r requirements.txt
python main.py
