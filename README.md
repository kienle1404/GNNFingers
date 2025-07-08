# GNNFingers: Verifying GNN Ownership via Fingerprinting

This project implements the paper **"GNNFingers: A Fingerprinting Framework for Verifying Ownerships of Graph Neural Networks"** (You et al., 2024). The goal is to detect unauthorized use of GNNs via graph fingerprinting without altering the original model.

## Structure
- `models/`: GCN, GAT, GIN implementations
- `data/`: Loaders for Cora, Citeseer, PubMed
- `fingerprints/`: Code to generate graph fingerprints
- `verifier/`: UniVerifier training and evaluation
- `utils/`: Utilities and helpers

## Getting Started

```bash
pip install -r requirements.txt
python main.py
