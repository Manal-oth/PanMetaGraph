"""Graph assembly and loading.

Stage A (offline, see scripts in the pipeline section of the README) produces a
single .pt per cohort holding the heterogeneous graph:

    x_dict            {node_type: FloatTensor[N_t, F_t]}
    node_type         LongTensor[N]        index into src.model.NODE_TYPES
    edge_index_dict   {relation: LongTensor[2, E_r]}
    edge_weight_dict  {relation: FloatTensor[E_r]}
    rw_landing        FloatTensor[N, k]    random-walk return probabilities
    y                 LongTensor[N_lab]    AMR family labels in [0, 541)
    label_index       LongTensor[N_lab]    node ids carrying a label
    split             {'train','val','test'} LongTensor index into label_index
    pair_index        LongTensor[2, P]     candidate transfer pairs
    pair_label        LongTensor[P]        weak labels for L_tr

`load_graph` is deliberately strict: it refuses to silently fabricate a graph,
so a missing artefact is an error rather than a run on random data.
"""
from pathlib import Path
import torch

REQUIRED = ["x_dict", "node_type", "edge_index_dict", "y", "label_index", "split"]


def graph_path(root, dataset):
    return Path(root) / dataset / f"{dataset}_graph.pt"


def load_graph(root, dataset, relations=None, drop_relations=(), use_rwpe=True,
               use_edge_weights=True, device="cpu"):
    p = graph_path(root, dataset)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Build it first:\n"
            f"  python download.py --dataset {dataset} --out {root}\n"
            f"  python build_graph.py --dataset {dataset} --root {root}"
        )
    g = torch.load(p, map_location=device, weights_only=False)
    missing = [k for k in REQUIRED if k not in g]
    if missing:
        raise KeyError(f"{p} is missing required keys: {missing}")

    ei = dict(g["edge_index_dict"])
    ew = dict(g.get("edge_weight_dict", {}))
    if relations is not None:
        ei = {r: v for r, v in ei.items() if r in relations}
        ew = {r: v for r, v in ew.items() if r in relations}
    for r in drop_relations:                     # ablations (ii), (iii), (xii)
        ei.pop(r, None)
        ew.pop(r, None)
    if not use_edge_weights:                     # ablation (ix)
        ew = {}

    batch = {
        "x_dict": g["x_dict"],
        "node_type": g["node_type"],
        "edge_index_dict": ei,
        "edge_weight_dict": ew,
        "rw_landing": g.get("rw_landing") if use_rwpe else None,   # ablation (i)
        "y_all": g["y"],
        "label_index_all": g["label_index"],
        "split": g["split"],
        "pair_index": g.get("pair_index"),
        "pair_label": g.get("pair_label"),
    }
    return batch


def split_view(batch, which):
    """Return a batch restricted to the labelled nodes of one split."""
    sel = batch["split"][which]
    out = dict(batch)
    out["label_index"] = batch["label_index_all"][sel]
    out["y"] = batch["y_all"][sel]
    return out


def node_batches(n, size, generator=None):
    """Yield index tensors of `size` labelled nodes (batch_size is in nodes)."""
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, size):
        yield perm[i:i + size]
