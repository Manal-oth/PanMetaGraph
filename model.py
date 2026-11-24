"""PanMetaGraph model: relation-specific graph transformer over a heterogeneous
pangenome-metagenome graph, with random-walk positional encoding and
transfer-probability refinement.

Implements Eqs. (8)-(19) of the manuscript.

Node types   (4): contig, gene_cluster, mge, sample
Relation types (4): contig-gene_cluster (ref alignment), gene_cluster-gene_cluster
                    (pangenome adjacency), *-mge (mobile genetic element),
                    sample-contig (ecological co-occurrence)
Task: 541-class node classification over AMR gene-family labels.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_TYPES = ["contig", "gene_cluster", "mge", "sample"]
RELATION_TYPES = ["ref_align", "pan_adj", "mge_link", "eco_cooc"]


class RandomWalkPE(nn.Module):
    """Eq. (8)-(9): random-walk positional encoding, k steps, projected by an MLP."""

    def __init__(self, k_steps: int, d_model: int):
        super().__init__()
        self.k_steps = k_steps
        self.mlp = nn.Sequential(
            nn.Linear(k_steps, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, rw_landing: torch.Tensor) -> torch.Tensor:
        # rw_landing: [N, k_steps] diagonal return probabilities of P^1..P^k
        return self.mlp(rw_landing)


class RelationAttention(nn.Module):
    """Eqs. (10)-(14): per-relation multi-head attention with its own Q/K/V and
    its own softmax normalisation, then aggregation across relations with a
    learnable scalar beta_r (Eq. 15)."""

    def __init__(self, d_model: int, n_heads: int, n_relations: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.n_relations = n_relations
        self.q = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(n_relations))
        self.k = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(n_relations))
        self.v = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(n_relations))
        self.beta = nn.Parameter(torch.zeros(n_relations))     # learnable beta_r
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _one_relation(self, r, x, edge_index, edge_weight):
        src, dst = edge_index
        N = x.size(0)
        q = self.q[r](x).view(N, self.n_heads, self.d_head)
        k = self.k[r](x).view(N, self.n_heads, self.d_head)
        v = self.v[r](x).view(N, self.n_heads, self.d_head)
        # Eq. (12): scaled dot-product score on each edge
        score = (q[dst] * k[src]).sum(-1) / math.sqrt(self.d_head)      # [E, H]
        if edge_weight is not None:                                     # Eq. (13)
            score = score + torch.log(edge_weight.clamp_min(1e-12)).unsqueeze(-1)
        # per-relation softmax over the in-neighbourhood of each dst node
        score = score - score.max()
        ex = score.exp()
        denom = torch.zeros(N, self.n_heads, device=x.device).index_add_(0, dst, ex)
        alpha = ex / denom[dst].clamp_min(1e-12)
        msg = alpha.unsqueeze(-1) * v[src]
        agg = torch.zeros_like(v).index_add_(0, dst, msg)
        return agg.reshape(N, self.d_model)

    def forward(self, x, edge_index_dict, edge_weight_dict):
        w = torch.softmax(self.beta, dim=0)                             # Eq. (15)
        h = torch.zeros_like(x)
        for r, rel in enumerate(RELATION_TYPES[: self.n_relations]):
            ei = edge_index_dict.get(rel)
            if ei is None or ei.numel() == 0:
                continue
            h = h + w[r] * self._one_relation(r, x, ei, edge_weight_dict.get(rel))
        return self.drop(self.out(h))


class EncoderLayer(nn.Module):
    """Eq. (16): pre-norm transformer block around the relation attention."""

    def __init__(self, d_model, n_heads, n_relations, ff_mult, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = RelationAttention(d_model, n_heads, n_relations, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ei, ew):
        x = x + self.attn(self.n1(x), ei, ew)
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class TransferHead(nn.Module):
    """Eqs. (17)-(18): transfer-probability refinement. Scores the plausibility of
    a horizontal transfer between two nodes and re-weights their embeddings."""

    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, h, pair_index):
        a, b = pair_index
        return torch.sigmoid(self.score(torch.cat([h[a], h[b]], dim=-1))).squeeze(-1)


class PanMetaGraph(nn.Module):
    def __init__(self, in_dims, d_model=256, n_heads=8, n_layers=4, n_classes=541,
                 n_relations=4, rw_steps=16, ff_mult=4, dropout=0.1, mu=0.1):
        super().__init__()
        self.mu = mu
        self.proj = nn.ModuleDict(
            {t: nn.Linear(in_dims[t], d_model) for t in NODE_TYPES if t in in_dims}
        )
        self.type_emb = nn.Embedding(len(NODE_TYPES), d_model)
        self.rwpe = RandomWalkPE(rw_steps, d_model)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, n_relations, ff_mult, dropout)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)
        self.transfer = TransferHead(d_model)

    def encode(self, x_dict, node_type, rw_landing, edge_index_dict, edge_weight_dict):
        N = node_type.size(0)
        d = self.type_emb.weight.size(1)
        h = torch.zeros(N, d, device=node_type.device)
        for ti, t in enumerate(NODE_TYPES):
            if t not in x_dict:
                continue
            m = node_type == ti
            if m.any():
                h[m] = self.proj[t](x_dict[t])
        h = h + self.type_emb(node_type)
        if rw_landing is not None:
            h = h + self.rwpe(rw_landing)                       # Eq. (9)
        for layer in self.layers:
            h = layer(h, edge_index_dict, edge_weight_dict)
        return self.norm(h)

    def forward(self, batch):
        h = self.encode(batch["x_dict"], batch["node_type"], batch.get("rw_landing"),
                        batch["edge_index_dict"], batch.get("edge_weight_dict", {}))
        logits = self.cls(h)
        out = {"logits": logits, "h": h}
        if batch.get("pair_index") is not None:
            out["p_transfer"] = self.transfer(h, batch["pair_index"])
        return out

    def loss(self, out, batch):
        """Eq. (19): L = L_cls + mu * L_tr."""
        idx = batch["label_index"]
        l_cls = F.cross_entropy(out["logits"][idx], batch["y"])
        total = l_cls
        l_tr = torch.zeros((), device=l_cls.device)
        if "p_transfer" in out and batch.get("pair_label") is not None:
            l_tr = F.binary_cross_entropy(out["p_transfer"], batch["pair_label"].float())
            total = total + self.mu * l_tr
        return total, {"loss": float(total), "l_cls": float(l_cls), "l_tr": float(l_tr)}
