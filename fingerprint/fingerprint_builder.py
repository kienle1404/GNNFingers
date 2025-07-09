import torch
from torch_geometric.data import Data

class FingerprintBuilder:
    """Handles graph-fingerprint init & gradient-based updates."""
    def __init__(self, num_nodes: int, feat_dim: int):
        self.num_nodes = num_nodes
        self.x = torch.randn(num_nodes, feat_dim, requires_grad=True)
        self.edge_index = self._init_sparse_edges()

    def _init_sparse_edges(self):
        # minimal random sparse E
        return torch.empty((2, 0), dtype=torch.long)

    def project(self):
        """Clip node features / discretize adjacency in-place."""
        self.x.data.clamp_(-1.0, 1.0)
        # add / remove edges according to stored gradient flags
