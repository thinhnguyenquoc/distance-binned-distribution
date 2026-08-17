"""
Urban Graph Neural Network Node Encoder.

Learns tract representation h_i from urban features X and spatial graph G^urban:
    h_i = GNN_theta(X, G^urban)

Graph structure:
    G^urban is built ONLY from observable spatial geography (k-NN / radius graph).
    No OD data is ever used to construct G^urban.

Architecture:
    Multi-layer Graph Convolution / GAT / GraphConv with residual connections and LayerNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Message passing layer with edge distance modulation.
    Aggregates neighbor features weighted by spatial distance:
        m_ij = W_msg * [h_j || log(1 + d_ij)]
        h_i' = W_self * h_i + Agg_{j in N(i)}(m_ij)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.msg_linear = nn.Linear(in_dim + 1, out_dim)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim)
        edge_index: (2, E_graph)
        edge_dist: (E_graph,) distance in km
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Log distance feature
        log_d = torch.log1p(edge_dist).unsqueeze(-1)  # (E_graph, 1)
        
        # Message computation: [h_src, log_d]
        msg_input = torch.cat([x[src], log_d], dim=-1)  # (E_graph, in_dim + 1)
        msg = self.msg_linear(msg_input)  # (E_graph, out_dim)

        # Scatter mean aggregation
        out = torch.zeros(x.size(0), msg.size(1), device=x.device, dtype=x.dtype)
        # Degree count for mean aggregation
        deg = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        
        out.index_add_(0, dst, msg)
        deg.index_add_(0, dst, torch.ones_like(log_d))
        
        out = out / torch.clamp(deg, min=1.0)
        
        # Combine with transformed self features
        h_self = self.self_linear(x)
        out = self.norm(F.relu(out + h_self))
        return out


class UrbanGNN(nn.Module):
    """
    Urban GNN Node Encoder that produces node embeddings h_i in R^d.
    """
    def __init__(
        self,
        in_dim: int = 26,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, in_dim) normalized node features.
            edge_index: (2, E_graph) spatial graph edges.
            edge_dist:  (E_graph,) geographic distances.

        Returns:
            h: (N, out_dim) node embeddings.
        """
        h = self.input_fc(x)
        for layer in self.layers:
            h_new = layer(h, edge_index, edge_dist)
            h = h + self.dropout(h_new)  # residual connection

        h = self.output_fc(h)
        return h


if __name__ == "__main__":
    gnn = UrbanGNN(in_dim=26, hidden_dim=32, out_dim=32, num_layers=2)
    x = torch.randn(10, 26)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    edge_dist = torch.tensor([1.0, 2.0, 1.5, 3.0, 0.5])
    h = gnn(x, edge_index, edge_dist)
    print(f"UrbanGNN output shape: {h.shape}")
