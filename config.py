"""Config loading with single-level `base:` inheritance, so each ablation file
only has to state what it changes."""
from pathlib import Path
import copy
import yaml


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path, overrides=None):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(load_config((path.parent / base).resolve()), cfg)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def _coerce(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def parse_set(pairs):
    """--set optim.lr=1e-4 model.d_model=128 -> nested dict"""
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        node = out
        parts = k.split(".")
        for q in parts[:-1]:
            node = node.setdefault(q, {})
        node[parts[-1]] = _coerce(v)
    return out
