# models/gnn_blocks.py
import os, sys, numpy as np, torch
import torch.nn as nn
from torch_geometric.nn import TAGConv, GCN2Conv, GATv2Conv

def build_edge_index_and_weight_from_ybus(ybus: np.ndarray, use_abs=True):
    N = ybus.shape[0]
    ii, jj = np.nonzero(np.triu(np.abs(ybus), k=1))
    if use_abs:
        w = np.abs(ybus[ii, jj]).astype(np.float32)
    else:
        w = (-np.real(ybus[ii, jj])).astype(np.float32)

    i2 = np.concatenate([ii, jj], axis=0)
    j2 = np.concatenate([jj, ii], axis=0)
    w2 = np.concatenate([w,  w ], axis=0)

    edge_index = torch.from_numpy(np.stack([i2, j2], axis=0)).long()   # [2, E]
    edge_weight = torch.from_numpy(w2).float()                         # [E]
    return edge_index, edge_weight, N

class GraphRefiner(nn.Module):
    def __init__(self, ybus: np.ndarray, state_dim: int,
                 gnn_type: str = "tag", hidden: int = 64, layers: int = 2, K: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        assert state_dim % 2 == 0
        self.N = state_dim // 2
        self.state_dim = state_dim
        self.node_feat_dim = 2
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout

        ei, ew, N = build_edge_index_and_weight_from_ybus(ybus)
        self.register_buffer("edge_index_single", ei)
        self.register_buffer("edge_weight_single", ew)
        assert N == self.N, "ybus 维度与 state_dim/2 不一致！"

        if self.gnn_type == "gcn2":
            # GCN2: 通道必须恒定 => 用线性层做 2↔hidden 的投影
            self.pre_lin  = nn.Linear(self.node_feat_dim, hidden)
            self.post_lin = nn.Linear(hidden, self.node_feat_dim)
            self.convs = nn.ModuleList([
                GCN2Conv(channels=hidden, alpha=0.1, theta=None,
                         shared_weights=True, cached=False, normalize=True)
                for _ in range(layers)
            ])
        else:
            # 其它卷积：2 -> hidden -> ... -> 2
            dims = [self.node_feat_dim, hidden] + [hidden]*(layers-2) + [self.node_feat_dim]
            self.convs = nn.ModuleList()
            for li in range(len(dims) - 1):   # ← 修复：用 len(dims)-1
                in_ch, out_ch = dims[li], dims[li+1]
                if self.gnn_type == "tag":
                    self.convs.append(TAGConv(in_ch, out_ch, K=K))
                elif self.gnn_type == "gat":
                    self.convs.append(GATv2Conv(in_ch, out_ch, heads=1, concat=True, dropout=dropout))
                else:
                    raise ValueError(f"Unknown gnn_type={gnn_type}")
        self.act = nn.ReLU()

    def _tile_edge_index(self, graphs: int, nodes_per_graph: int, device):
        ei = self.edge_index_single.to(device); ew = self.edge_weight_single.to(device)
        E = ei.size(1)
        offset = torch.arange(graphs, device=device) * nodes_per_graph
        offset = offset.view(-1, 1).repeat(1, E).reshape(-1)
        ei_rep = ei.repeat(1, graphs) + offset.unsqueeze(0)
        ew_rep = ew.repeat(graphs)
        return ei_rep, ew_rep

    def forward(self, x):
        device = x.device
        N = self.N; F = self.node_feat_dim
        shp = x.shape; rank = x.dim()

        if rank == 3 and x.size(-1) == 2*N:         # (B,W,2N)
            B, W, _ = shp; G = B * W
            x_nodes = x.view(B, W, N, F)
            x_flat  = x_nodes.reshape(G * N, F)
            back = ("BW", B, W)
        elif rank == 3 and x.size(-1) == F:         # (G,N,2)
            G, Nin, Fin = shp; assert Nin == N and Fin == F
            x_flat = x.reshape(G * N, F)
            back = ("GN", G, None)
        elif rank == 2 and x.size(-1) == F:         # (G*N,2)
            L = x.size(0); assert L % N == 0
            G = L // N
            x_flat = x
            back = ("LN", G, None)
        else:
            raise RuntimeError(f"Unsupported input shape={shp}")

        ei_rep, ew_rep = self._tile_edge_index(G, N, device)

        if self.gnn_type == "gcn2":
            h = self.pre_lin(x_flat)
            x0 = h.clone()
            for i, conv in enumerate(self.convs):
                h = conv(h, x0, ei_rep, edge_weight=ew_rep)
                if i < len(self.convs) - 1:
                    h = self.act(h)
                    if self.dropout > 0:
                        h = nn.functional.dropout(h, p=self.dropout, training=self.training)
            h = self.post_lin(h)  # 回到 2 维
        else:
            h = x_flat
            for i, conv in enumerate(self.convs):
                if self.gnn_type == "tag":
                    h = conv(h, ei_rep, ew_rep)
                elif self.gnn_type == "gat":
                    h = conv(h, ei_rep)
                if i < len(self.convs) - 1:
                    h = self.act(h)
                    if self.dropout > 0:
                        h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        # 还原 & 残差（此时 h 的最后一维一定是 2）
        if back[0] == "BW":
            B, W = back[1], back[2]
            h_nodes = h.view(B, W, N, F)
            out = x.view(B, W, N, F) + h_nodes
            out = out.reshape(B, W, 2*N)
        elif back[0] == "GN":
            G = back[1]
            h_nodes = h.view(G, N, F)
            out = x + h_nodes
        else:  # "LN"
            out = x + h
        return out
